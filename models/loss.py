import tensorflow as tf
import tensorflow.keras.backend as K
from src.postprocessing import bbox_iou, decode_model_outputs

def lossFunction(anchors, ignore_thresh=0.5, lambda_box = 5, lambda_noobj = 0.5, lambda_class = 1):

    """
    Wrapper to compute the loss function
    
    Inputs: anchors - [BS, W, H]
    
    """

    def loss(y_true, y_pred): 
        """
        y_true: [BS, grid_size, grid_size, num_anchors, (y, x, h, w, p, obj_class)] - output of TransformBoxes.bboxes_to_grid()
        
        y_pred: [BS, gridy, gridx, NUM_ANCHORS, 4 + 1 + NUM_CLASSES] (y, x, h, w, p, NUM_CLASSES)

        y, x, h, w are values between 0 and 1
        """
        
        # Split y_pred
        pred_bbox, pred_confidence, pred_obj_class, pred_cell_box = decode_model_outputs(y_pred, anchors) # [BS, gy, gx, NUM_ANCHORS, 4], [BS, gy, gx, NUM_ANCHORS, 1], [BS, gy, gx, NUM_ANCHORS, NUM_CLASSES], [BS, gy, gx, NUM_ANCHORS, 4]
        pred_yx = pred_cell_box[..., :2]
        pred_hw = pred_cell_box[..., 2:]

        # Split y_true and get scale to give higher weight to smaller boxes
        true_yxhw, true_confidence, true_obj_class = tf.split(y_true, (4, 1, 1), axis=-1) # [BS, gy, gx, anchors, (y, x, h, w)], [BS, gy, gx, anchors, 1], [BS, gy, gx, anchors, 1]
        true_yx = true_yxhw[..., :2] # Values between 0 and 1 to indicate where they are locally located within a cell
        true_hw = true_yxhw[..., 2:] # Values between 0 and 1 to indicate local size within a cell
        bbox_loss_scale = 2.0 - 1.0 * true_hw[..., 0] * true_hw[..., 1]

        # Convert from information within grid cell to global image y, x, h, w values 
        grid_size = tf.shape(y_true)[1:3]
        grid = tf.meshgrid(tf.range(grid_size[0]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2) # [gy, gx, 1, 2]
        global_yx = (true_yx + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        global_hw = tf.math.log(1e-16 + true_hw / anchors[..., ::-1])
        true_bbox = tf.concat([global_yx, global_hw], axis=-1)

        # Get Object mask
        obj_mask = tf.squeeze(true_confidence, axis=-1) # [BS, gy, gx, anchors]

        # Get Ignore Mask: 1 - when iou is below threshold 0 - else
        boxes1_yxyx = tf.concat([pred_bbox[..., :2] - pred_bbox[..., 2:] * 0.5, pred_bbox[..., :2] + pred_bbox[..., 2:] * 0.5], axis=-1) # [BS, gy, gx, anchor, (yxyx)]
        boxes2_yxyx = tf.concat([true_bbox[..., :2] - true_bbox[..., 2:] * 0.5, true_bbox[..., :2] + true_bbox[..., 2:] * 0.5], axis=-1) # [BS, gy, gx, anchor, (yxyx)]
        iou = bbox_iou(boxes1_yxyx, boxes2_yxyx) # [BS, gy, gx, anchor, anchor]
        max_iou = tf.reduce_max(iou, axis=-1) # [BS, gy, gx, anchor]
        ignore_mask = tf.cast(max_iou < ignore_thresh, tf.float32) # [BS, gy, gx, anchor]

        # Get Regression Loss
        yx_loss = obj_mask * bbox_loss_scale * tf.reduce_sum(tf.square( true_yx - pred_yx ), axis=-1) # out: [BS, gy, gx, anchor]
        hw_loss = obj_mask * bbox_loss_scale * tf.reduce_sum(tf.square( tf.sqrt(tf.abs(true_hw)) - tf.sqrt(tf.abs(pred_hw)) ), axis=-1) # out: [BS, gy, gx, anchor]

        # Get Confidence Loss
        conf_loss = tf.squeeze(K.binary_crossentropy(true_confidence, pred_confidence)) # out: [BS, gy, gx, anchor]
        conf_loss =  obj_mask * conf_loss + lambda_noobj * (1.-obj_mask) * ignore_mask * conf_loss

        # Get Classification Loss
        class_loss = obj_mask * K.sparse_categorical_crossentropy(true_obj_class, pred_obj_class) # out: [BS, gy, gx, anchor]

        yx_loss = tf.reduce_sum(yx_loss, axis=(1, 2, 3))
        hw_loss = tf.reduce_sum(hw_loss, axis=(1, 2, 3))
        conf_loss = tf.reduce_sum(conf_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        
        return lambda_box * yx_loss + lambda_box * hw_loss + conf_loss + lambda_class * class_loss
    return loss