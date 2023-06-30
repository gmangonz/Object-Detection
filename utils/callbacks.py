from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from utils.utils import visualize_boxes
import warnings

class DisplayCallback(tf.keras.callbacks.Callback):
    
  def __init__(self, 
               img_path, 
               args, 
               anchors,
               **kwargs):
    
    super(DisplayCallback, self).__init__(**kwargs)
    self.img = img_to_array(load_img(img_path, target_size=args.img_size)) / 255
    self.scale = tf.constant([args.img_size[0], args.img_size[1], args.img_size[0], args.img_size[1]], dtype=tf.float32)
    self.anchors = anchors

  def on_epoch_begin(self, epoch, logs=None):
    
    if (epoch + 1) % 3 == 0 or epoch == 0:

        predicted = self.model.model(self.img[None, ...], training=False)

        for i in range(self.anchors.shape[0]):
            bboxes, _, _ = self.model.post_process(predicted[i], self.anchors[i])
            visualize_boxes(self.img, tf.squeeze(bboxes)*self.scale, figsize=(5, 5), linewidth=1, color=[0, 0, 1])

class SaveModel(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, 
                 model_to_save, 
                 **kwargs):
        
        super(SaveModel, self).__init__(**kwargs)
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                                'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best} to {current}, saving model to {filepath}')
                    self.best = current
                    if self.save_weights_only:
                        self.model_to_save.save_weights(filepath, overwrite=True)
                    else:
                        self.model_to_save.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best}')
        else:
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1}: saving model to {filepath}')
            if self.save_weights_only:
                self.model_to_save.save_weights(filepath, overwrite=True)
            else:
                self.model_to_save.save(filepath, overwrite=True)

        super(SaveModel, self).on_batch_end(epoch, logs)


def create_callbacks(filepath_name, args, model_to_save, anchors, img_path = r'D:\DL-CV-ML Projects\Turion_Space\Updated_Turion_Space\imgs\img.png'):
    
    early_stop = EarlyStopping(
        monitor              = args.monitor, 
        min_delta            = 0.01, 
        patience             = 7, 
        mode                 = 'auto', 
        restore_best_weights = True
    )

    checkpoint = SaveModel(
        model_to_save      = model_to_save,
        filepath           = filepath_name,
        monitor            = args.monitor,
        verbose            = 1, 
        save_best_only     = True, 
        save_weights_only  = True,
        mode               = 'min', 
        save_freq          = 'epoch'
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor    = args.monitor,
        factor     = 0.1,
        patience   = 5,
        verbose    = 1,
        mode       = 'min',
        min_delta  = 0.01,
        cooldown   = 0,
        min_lr     = 0
    )

    display = DisplayCallback(
        img_path = img_path,
        args     = args,
        anchors  = anchors
    )

    return [early_stop, checkpoint, reduce_on_plateau, display]