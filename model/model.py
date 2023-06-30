import tensorflow as tf
from tensorflow.keras import layers
from models.bbox_preprocess import TransformBoxes
from utils.postprocess.postprocessing import bbox_iou, decode_model_outputs, non_max_suppression

class Mean(tf.keras.metrics.Mean):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, sample_weight=sample_weight)


def make_dummy_model(img_size, grid_sizes, classes = 100, num_anchors = 3):

    if len(img_size) == 2:
        shape = (img_size[0], img_size[1], 3)
    if len(img_size) == 3:
        shape = img_size
    
    inputs = tf.keras.Input(shape=shape)
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding="same", name='Conv1')(inputs)
    x = layers.BatchNormalization(name='BN1')(x)
    x = layers.Activation("relu", name='Act1')(x)

    x = layers.Conv2D(32, kernel_size=3, strides=4, padding="same", name='Conv2')(x)
    x = layers.BatchNormalization(name='BN2')(x)
    x = layers.Activation("relu", name='Act2')(x)

    x = layers.Conv2D(32, kernel_size=3, strides=4, padding="same", name='Conv3')(x)
    x = layers.BatchNormalization(name='BN3')(x)
    x = layers.Activation("relu", name='Act3')(x)

    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", name='Conv4')(x)
    x = layers.BatchNormalization(name='BN4')(x)
    x = layers.Activation("relu", name='Act4')(x)

    filters = num_anchors * (4 + 1 + classes)
    x = layers.Conv2D(filters=filters, kernel_size=1, padding="same", name='FinalConv1')(x)
    out1 = layers.Reshape((grid_sizes[0], grid_sizes[0], num_anchors, classes + 5))(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(filters=filters, kernel_size=1, padding="same", name='FinalConv2')(x)
    out2 = layers.Reshape((grid_sizes[1], grid_sizes[1], num_anchors, classes + 5))(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(filters=filters, kernel_size=1, padding="same", name='FinalConv3')(x)
    out3 = layers.Reshape((grid_sizes[2], grid_sizes[2], num_anchors, classes + 5))(x)

    model = tf.keras.Model(inputs=inputs, outputs=[out1, out2, out3])
    return model


class MyModel(tf.keras.Model):

    def __init__(self, 
                 img_size,
                 augment_func,
                 grid_sizes,
                 anchors,
                 args,
                 **kwargs):
               
        super(MyModel, self).__init__(**kwargs)

        self.img_size = img_size
        self.train_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.model = make_dummy_model(img_size, grid_sizes, classes = args.NUM_CLASSES, num_anchors = 3)
        self.transformBoxes = TransformBoxes(args=args, grid_sizes=grid_sizes, anchors=anchors)
        self.args = args
        self.anchors = anchors
        self.ada = augment_func

    def compile(self, optimizer=None, metrics=[], *args, **kwargs):

        assert isinstance(metrics, list), "metrics must be a list"
        self.train_step_counter.assign(0)
        self.optimizer = optimizer
        self.augmentation_probability_tracker = tf.keras.metrics.Mean(name="aug_probability")
        self.iou_tracker = tf.keras.metrics.Mean(name="iou_tracker")
        super(MyModel, self).compile(metrics=metrics, *args, **kwargs)

    def post_process(self, predicted, anchors):
        
        # Get bboxes predicted out: [BS, gy, gx, NUM_ANCHORS, (y, x, h, w)], [BS, gy, gx, NUM_ANCHORS, 1], [BS, gy, gx, NUM_ANCHORS, NUM_CLASSES]
        pred_bbox, pred_confidence, pred_obj_class, _ = decode_model_outputs(predicted, anchors)

        # NMS out: [BS, max_output_size, (y, x, y, x)], [BS, max_output_size], [BS, max_output_size]
        boxes, scores, classes, _ = non_max_suppression(pred_bbox, pred_confidence, pred_obj_class, self.args.NUM_CLASSES, self.args.max_output_size, self.args.max_output_size_per_class, self.args.iou_threshold, self.args.confidence_threshold)
        return boxes, scores, classes

    def get_iou(self, true_bboxes, true_mask, predicted, anchors):

        boxes, _, _ = self.post_process(predicted, anchors)
        iou = bbox_iou(true_bboxes, boxes) * true_mask # IOU out: [BS, N, max_output_size]
        return tf.reduce_mean(iou, axis=(2, 1, 0))

    def train_step(self, ds_input): # image (BS, H, W, C), bboxes (BS, N, 5)

        self.train_step_counter.assign_add(1)

        # Augment images and get mask
        augmented_images, augmented_bboxes = self.ada(ds_input, training=True) # augmented_images - [BS, H, W, C], augmented_bboxes - [BS, N, (y1, x1, y2, x2, obj_class)]
        true_mask = tf.math.count_nonzero(augmented_bboxes, axis=-1,  dtype=tf.bool)[..., None]
        true_mask = tf.cast(true_mask, tf.float32) # (BS, N, 1)

        # Trasform bboxes to grid inputs
        grid_aug_bboxes = self.transformBoxes(augmented_bboxes) # out: [BS, gy, gx, num_anchors, (y, x, h, w, p, obj_class)] x 3
        
        # Gradient
        with tf.GradientTape() as tape:
            
            # Get predicted and loss
            predicted = self.model(augmented_images, training=True) # out: [BS, gy, gx, num_anchors, (y, x, h, w, p, NUM_CLASSES)] x 3
            loss = self.compiled_loss(grid_aug_bboxes, predicted)
        
        # Optimize
        trainable_weights = self.model.trainable_weights
        model_grads = tape.gradient(loss, trainable_weights)
        self.optimizer.apply_gradients(zip(model_grads, trainable_weights))

        # Get Resulting IOU
        iou1 = self.get_iou(augmented_bboxes[..., :-1], true_mask, predicted[0], self.anchors[0])
        iou2 = self.get_iou(augmented_bboxes[..., :-1], true_mask, predicted[1], self.anchors[1])
        iou3 = self.get_iou(augmented_bboxes[..., :-1], true_mask, predicted[2], self.anchors[2])
        total_iou = (iou1 + iou2 + iou3) / 3.

        # Update metrics
        self.ada.update(1. - total_iou) # Why 1 - iou? Because inside Ada: accuracy = K.mean(1. - loss)
        self.compiled_metrics.update_state([loss], [loss])
        self.augmentation_probability_tracker.update_state(self.ada.probability)
        self.iou_tracker.update_state(total_iou)
        
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False, augment=False):

        augmented_images, augmented_masks = self.ada(inputs, training=training) if augment else inputs
        predicted = self.model(augmented_images, training=training)
        return augmented_images, augmented_masks, predicted