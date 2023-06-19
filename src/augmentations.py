from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import numpy as np

class Converter(layers.Layer):

  """
  Base class to use for all data augmentation layers
  """  

  def __init__(self, **kwargs):
  
    super(Converter, self).__init__(**kwargs)

  def Un_Normalize(self, bboxes_normalized, shape):
      """
      Inputs: 
        bboxes_normalized - bounding boxes with values between 0 and 1
          shape - (n, 4)
        
        shape - shape of the image to use
          shape - (2,) or (3,)
      
      Returns:
        bboxes - bounding boxes with values between (0, 0) and IMG_SIZE
      
      """

      shape = tf.cast(shape, dtype=bboxes_normalized.dtype)

      bboxes = tf.stack([bboxes_normalized[:, 0] * shape[0], 
                         bboxes_normalized[:, 1] * shape[1],
                         bboxes_normalized[:, 2] * shape[0],
                         bboxes_normalized[:, 3] * shape[1]], axis=-1)
      return bboxes

  def Normalize(self, bboxes, shape):
    
      """
      Inputs: 
        bboxes - bounding boxes with values between (0, 0) and IMG_SIZE
          shape - (n, 4)
        
        shape - shape of the image to use
          shape - (2,) or (3,)
      
      Returns:
        
        bboxes_normalized - bounding boxes with values between 0 and 1
        
      """

      shape = tf.cast(shape, dtype=bboxes.dtype)
      bboxes_normalized = tf.stack([bboxes[:, 0] / shape[0], 
                                    bboxes[:, 1] / shape[1], 
                                    bboxes[:, 2] / shape[0], 
                                    bboxes[:, 3] / shape[1]], axis=-1)
      return bboxes_normalized

  def yxyx_to_yxhw(self, bboxes):
    """
    Input:
      bboxes: y1, x1, y2, x2
    
    Returns:
      yxhw: y, x, h, w - calcualted by [(y1, x1) + (y2, x2)] / 2 and [(y2, x2) - (y1, x1)]
    
    """

    return tf.concat( [ (bboxes[..., :2] + bboxes[..., 2:]) / 2.0, bboxes[..., 2:] - bboxes[..., :2] ], axis=-1 ) # y, x, h, w

  def yxhw_to_yxyx(self, bboxes):

      """
      Input:
        bboxes: y_center, x_center, h, w 

      Returns:
        yxyx: bounding boxes containing y1, x1, y2, x2 
      
      """

      return tf.concat( [ bboxes[..., :2] - bboxes[..., 2:] / 2.0, bboxes[..., :2] + bboxes[..., 2:] / 2.0 ], axis=-1 )
  
  def mask(self, bboxes):
      
      """
      Input: 
        bboxes: [bs, N, 4] - ymin, xmin, ymax, xmax 
    
      Returns:
        mask: [bs, N, 4] where the N rows are filled with zeros where there were originally no bounding boxes

      """
      mask = tf.math.count_nonzero(bboxes, axis=-1, dtype=tf.bool, keepdims=True)
      return tf.repeat(tf.cast(mask, dtype=tf.float32), 4, axis=-1)
  
  def bbox_area(self, bboxes, img_size):
      
      """
      Input:
        bboxes: Normalized bboxs btwn 0-1 - [bs, N, 4] or [N, 4] 
    
      Returns:
        area: [bs, N,] or [N,] calculated area of the bounding boxes
      """
      
      area = (bboxes[...,2] - bboxes[...,0])*(bboxes[...,3] - bboxes[...,1])
      area = area * img_size[0] * img_size[1]
      return area
  
  def remove_those_with_area_under_thresh(self, threshold, original_bboxes, aug_bboxes, img_size):
      
      """
      After data augmentation, some of the bounding boxes may have experience a significant change in their area (e.g. moved outside of the image), remove those
      The shapes of original_bboxes, aug_bboxes SHOULD be the same

      Input:
        threshold: threshold to determine if there was a significant change
        original_bboxes: [bs, N, 4] or [N, 4] 
        aug_bboxes: [bs, N, 4] or [N, 4] 
    
      Returns:
        aug_bboxes: [bs, N, 4] or [N, 4]
      """
      zero_bboxes = tf.zeros_like(original_bboxes)
      original_area = self.bbox_area(original_bboxes, img_size)
      aug_area = self.bbox_area(aug_bboxes, img_size)
      ratio = aug_area / original_area
      return tf.where(tf.math.less(ratio[..., None], threshold), zero_bboxes, aug_bboxes)


class RandomHorizontalFlip(Converter):

    """Flips the image upside down"""

    def __init__(self, 
                 probability=None, 
                 **kwargs):
        
        self.probability = probability
        super(RandomHorizontalFlip, self).__init__(**kwargs)
    
    def flip_imgs(self, imgs): 

        """Flipping upside down means flipping along the x axis"""
        flipped_img = imgs[:, ::-1, :, :]
        return flipped_img

    def flip_bboxes(self, bboxes):
        
        bbox_shape = tf.shape(bboxes) # [bs, n, 4]
        updates = 1. - bboxes[:, :, 2::-2] # [bs, n, 2] 
        updates = tf.transpose(updates, [2, 0, 1]) # [2, bs, n]
        indices = tf.constant([[0], [2]])
        bboxes_y = tf.scatter_nd(indices=indices, updates=updates, shape=tf.roll(bbox_shape, shift=1, axis=0)) # [4, bs, n]
        bboxes_y = tf.transpose(bboxes_y, [1, 2, 0]) # [bs, n, 4]
        bboxes_x = bboxes * tf.constant([[[0., 1., 0., 1.]]]) 

        flipped_bboxes = bboxes_x + bboxes_y
        mask = self.mask(bboxes)
        return flipped_bboxes * mask

    def get_batch_wise(self, inputs):

        imgs, bboxes, rand = inputs[0], inputs[1], inputs[2]
        batch_size = tf.shape(imgs)[0]
        augmentation_values = tf.random.uniform(shape=(batch_size,), minval=0.0, maxval=1.0)
        augmentation_bools = tf.math.less(augmentation_values, rand)
        imgs = tf.where(augmentation_bools[..., None, None, None], self.flip_imgs(imgs), imgs)
        bboxes = tf.where(augmentation_bools[..., None, None], self.flip_bboxes(bboxes), bboxes)
        return imgs, bboxes
    
    def call(self, inputs):
        
        return self.get_batch_wise(inputs)


class RandomMirror(Converter):

    """Flips the image left and right"""

    def __init__(self, 
                 probability=None, 
                 **kwargs):
        
        self.probability = probability
        super(RandomMirror, self).__init__(**kwargs)
    
    def mirror_imgs(self, img):
      
        mirror_img = img[:, :, ::-1, :]
        return mirror_img

    def mirror_bboxes(self, bboxes):

        """Flipping upside down means flipping along the y axis"""

        bbox_shape = tf.shape(bboxes) # [bs, n, 4]
        updates = 1. - bboxes[:, :, 3::-2] # [bs, n, 2] 
        updates = tf.transpose(updates, [2, 0, 1]) # [2, bs, n]
        indices = tf.constant([[1], [3]])
        bboxes_x = tf.scatter_nd(indices=indices, updates=updates, shape=tf.roll(bbox_shape, shift=1, axis=0)) # [4, bs, n]
        bboxes_x = tf.transpose(bboxes_x, [1, 2, 0]) # [bs, n, 4]
        bboxes_y = bboxes * tf.constant([[[1., 0., 1., 0.]]]) 

        mirror_bboxes = bboxes_x + bboxes_y
        mask = self.mask(bboxes)
        return mirror_bboxes * mask
    
    def get_batch_wise(self, inputs):

        imgs, bboxes, rand = inputs[0], inputs[1], inputs[2]
        batch_size = tf.shape(imgs)[0]
        augmentation_values = tf.random.uniform(shape=(batch_size,), minval=0.0, maxval=1.0)
        augmentation_bools = tf.math.less(augmentation_values, rand)
        imgs = tf.where(augmentation_bools[..., None, None, None], self.mirror_imgs(imgs), imgs)
        bboxes = tf.where(augmentation_bools[..., None, None], self.mirror_bboxes(bboxes), bboxes)
        return imgs, bboxes

    def call(self, inputs):
        
        return self.get_batch_wise(inputs)


class RandomZoom(Converter):

    def __init__(self, 
                 threshold = 0.3,
                 probability=None, 
                 scale=0.25, 
                 diff=True, 
                 **kwargs):
        
        self.probability = probability
        self.scale = scale
        self.diff = diff
        self.threshold = threshold
        super(RandomZoom, self).__init__(**kwargs)
    
    def get_scale_xy_diff(self, scale, dtype):

        scale_xy = tf.random.uniform(shape=(2,), minval=scale[0], maxval=scale[1], dtype=dtype) # Get values btwn 0-1 to use to scale x and y respectively
        scale_x = scale_xy[0]
        scale_y = scale_xy[1]
        return scale_x, scale_y
    
    def get_scale_xy_same(self, scale, dtype):

        scale_x = tf.random.uniform(shape=(1,), minval=scale[0], maxval=scale[1], dtype=dtype) # Get value btwn 0-1 to use to scale both x and y
        scale_y = scale_x
        return scale_x, scale_y

    def zoom(self, inputs):

        img, bboxes_in = inputs # img: h, w, 3 bboxes: n, 4
        img_shape = tf.cast(tf.shape(img), dtype=bboxes_in.dtype) # h, w, c
        bboxes = self.Un_Normalize(bboxes_in, img_shape) # MAY NEED TO CHECK THIS WITH COCO

        # Get scales
        scale = (tf.math.maximum(-1., -self.scale), tf.constant(self.scale))
        scale_x, scale_y = tf.cond(tf.constant(self.diff), lambda: self.get_scale_xy_diff(scale, bboxes.dtype), lambda: self.get_scale_xy_same(scale, bboxes.dtype))

        # Scale image
        resize_scale_y = tf.cast( (1 + scale_y) * img_shape[0], tf.int32) # New y size
        resize_scale_x = tf.cast( (1 + scale_x) * img_shape[1], tf.int32) # New x size
        resized_image = tf.image.resize(tf.convert_to_tensor(img), (resize_scale_y, resize_scale_x), method='nearest') # Resize
        resized_shape = tf.cast(tf.shape(resized_image), dtype=bboxes.dtype) # bs, new_h, new_w, 3 

        # Pad the image and crop if needed
        pad_y = tf.math.maximum(img_shape[0] - resized_shape[0], 0) # How much to pad height
        pad_y_before = pad_y//2
        pad_x = tf.math.maximum(img_shape[1] - resized_shape[1], 0) # How much to pad width
        pad_x_before = pad_x//2
        paddings = [[pad_y_before, pad_y - pad_y_before], [pad_x_before, pad_x - pad_x_before], [0, 0]]
        resized_image = tf.pad(resized_image, paddings, "REFLECT") # Pad image to maintain original size
        resized_image = resized_image[:tf.cast(img_shape[0], dtype=tf.int32),:tf.cast(img_shape[1], dtype=tf.int32),:] # Crop if needed

        # repeat [1+scale_y, 1+scale_x, 1+scale_y, 1+scale_x] n times where n is the number of bounding boxes seen
        bbox_scale = tf.tile([[(1 + scale_y), (1 + scale_x), (1 + scale_y), (1 + scale_x)]], [tf.shape(bboxes)[0], 1])
        bbx_scaled = bboxes*tf.cast(bbox_scale, bboxes.dtype) # Resize bboxes

        # Crop the bounding boxes if needed
        clip_box = [0, 0, 1+img_shape[0], 1+img_shape[1]]
        y_min = tf.reshape(tf.math.maximum(bbx_scaled[:,0], clip_box[0]), [-1, 1]) + pad_y_before
        x_min = tf.reshape(tf.math.maximum(bbx_scaled[:,1], clip_box[1]), [-1, 1]) + pad_x_before
        y_max = tf.reshape(tf.math.minimum(bbx_scaled[:,2], clip_box[2]), [-1, 1]) + pad_y_before
        x_max = tf.reshape(tf.math.minimum(bbx_scaled[:,3], clip_box[3]), [-1, 1]) + pad_x_before
        bbox_resized = tf.concat([y_min, x_min, y_max, x_max], axis=1) # bboxes: y1, x1, y2, x2 
        bbox_out = self.Normalize(bbox_resized, img_shape)

        # Postprocess Masking
        mask = self.mask(bboxes)
        bbox_out = bbox_out * mask
        bbox_out = self.remove_those_with_area_under_thresh(self.threshold, bboxes_in, bbox_out, img_shape)
        return resized_image, bbox_out

    def get_batch_wise(self, inputs):

        img, bboxes, rand = inputs[0], inputs[1], inputs[2]
        augment_value = tf.random.uniform(shape=tf.shape(rand))
        return tf.cond(tf.math.less(augment_value, rand), lambda: self.zoom((img, bboxes)), lambda: (img, bboxes))

    def call(self, inputs):
        
        output_sign = (tf.TensorSpec(shape=inputs[0].shape[1:], dtype=inputs[0].dtype), tf.TensorSpec(shape=inputs[1].shape[1:], dtype=inputs[1].dtype))
        return tf.map_fn(self.get_batch_wise, elems=inputs, fn_output_signature = output_sign)


class RandomTranslate(Converter):

    def __init__(self, 
                threshold = 0.3, 
                probability=None, 
                translate=0.2, 
                diff=True, 
                **kwargs):

        super(RandomTranslate, self).__init__(**kwargs)
        self.probability = probability
        self.translate = translate
        assert self.translate > 0 and self.translate < 1
        self.diff = diff
        self.threshold = threshold

    def translate_inputs(self, inputs):

        img, bboxes = inputs[0], inputs[1]
        translate = (-self.translate, self.translate)
        img_shape = tf.shape(img) # h, w, c
        img_shape = tf.cast(img_shape, dtype=bboxes.dtype)

        translate_xy = tf.random.uniform(shape=(2,), minval=translate[0], maxval=translate[1]) # Get values to use to translate x and y respectively 
        translate_x = translate_xy[0]
        translate_y = translate_xy[1]

        # Translate Image
        img_translate = tfa.image.translate(img, [-translate_x*img_shape[0], -translate_y*img_shape[1]], interpolation='nearest', fill_mode='nearest')

        bbx_transform = tf.tile([tf.concat([[translate_y, translate_x], [translate_y, translate_x]], axis=0)], [tf.shape(bboxes)[0], 1]) # ty, tx, ty, tx
        bbx_translated = bboxes - bbx_transform # subract because that what apply_affine_transform does

        # Crop the bboxes if needed
        clip_box = [0, 0, 1, 1]
        y_min = tf.reshape(tf.math.maximum(bbx_translated[:,0], clip_box[0]), [-1, 1])
        x_min = tf.reshape(tf.math.maximum(bbx_translated[:,1], clip_box[1]), [-1, 1])
        y_max = tf.reshape(tf.math.minimum(bbx_translated[:,2], clip_box[2]), [-1, 1])
        x_max = tf.reshape(tf.math.minimum(bbx_translated[:,3], clip_box[3]), [-1, 1])
        
        bbx_translated = tf.concat([y_min, x_min, y_max, x_max], axis=1) # bboxes: y1, x1, y2, x2

        # Postprocess Masking
        mask = self.mask(bboxes)
        bbx_translated = bbx_translated * mask
        bbx_translated = self.remove_those_with_area_under_thresh(self.threshold, bboxes, bbx_translated, img_shape)
        return img_translate, bbx_translated

    def get_batch_wise(self, inputs):

        img, bboxes, rand = inputs[0], inputs[1], inputs[2]
        augment_value = tf.random.uniform(shape=tf.shape(rand))
        return tf.cond(tf.math.less(augment_value, rand), lambda: self.translate_inputs((img, bboxes)), lambda: (img, bboxes))

    def call(self, inputs):
        
        output_sign = (tf.TensorSpec(shape=inputs[0].shape[1:], dtype=inputs[0].dtype), tf.TensorSpec(shape=inputs[1].shape[1:], dtype=inputs[1].dtype))
        return tf.map_fn(self.get_batch_wise, elems=inputs, fn_output_signature = output_sign)


class AddNoise(Converter):

    def __init__(self, 
                 probability=None, 
                 **kwargs):

        super(AddNoise, self).__init__(**kwargs)
        self.probability = probability

    def add_noise(self, inputs):
        
        img, bboxes = inputs[0], inputs[1]
        noise = tf.random.normal(tf.shape(img), mean=0, stddev=0.075, dtype=img.dtype)
        img_noise = tf.clip_by_value(img + noise, 0.0, 1.0)
        img_noise = tf.cast(img_noise, img.dtype)
        return img_noise, bboxes

    def get_batch_wise(self, inputs):

        img, bboxes, rand = inputs[0], inputs[1], inputs[2]
        augment_value = tf.random.uniform(shape=tf.shape(rand))
        return tf.cond(tf.math.less(augment_value, rand), lambda: self.add_noise((img, bboxes)), lambda: (img, bboxes))

    def call(self, inputs):
        
        output_sign = (tf.TensorSpec(shape=inputs[0].shape[1:], dtype=inputs[0].dtype), tf.TensorSpec(shape=inputs[1].shape[1:], dtype=inputs[1].dtype))
        return tf.map_fn(self.get_batch_wise, elems=inputs, fn_output_signature = output_sign)


class GaussianBlur(Converter):

    def __init__(self,
                 probability=None, 
                 kernel_size=3, 
                 sigma=1, 
                 **kwargs):
        super(GaussianBlur, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.probability = probability

    def gaussian_kernel(self, size=3, sigma=1):

        x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
        y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

        xs, ys = tf.meshgrid(x_range, y_range)
        r = tf.cast(xs**2 + ys**2, tf.float32)
        exp = tf.exp(-(r)/(2*(sigma**2)))
        kernel = exp / (2*np.pi*(sigma**2))
        return tf.cast( kernel / tf.reduce_sum(kernel), tf.float32)

    def blur_image(self, img):
       
        kernel = self.gaussian_kernel(self.kernel_size, self.sigma)
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1) # filter/kernel should be a tensor of shape [filter_height, filter_width, in_channels, out_channels]
        
        r, g, b = tf.split(img, [1,1,1], axis=-1)
        r_blur = tf.nn.conv2d(r[None, ...], filters=kernel, strides=[1,1,1,1], padding='SAME', name='r_blur')
        g_blur = tf.nn.conv2d(g[None, ...], filters=kernel, strides=[1,1,1,1], padding='SAME', name='g_blur')
        b_blur = tf.nn.conv2d(b[None, ...], filters=kernel, strides=[1,1,1,1], padding='SAME', name='b_blur')

        blur_image = tf.concat([r_blur[0], g_blur[0], b_blur[0]], axis=-1)
        blur_image = tf.cast(blur_image, img.dtype)

        return blur_image

    def get_batch_wise(self, inputs):

        img, bboxes, rand = inputs[0], inputs[1], inputs[2]
        augment_value = tf.random.uniform(shape=tf.shape(rand))
        return tf.cond(tf.math.less(augment_value, rand), lambda: (self.blur_image(img), bboxes), lambda: (img, bboxes))

    def call(self, inputs):
        
        output_sign = (tf.TensorSpec(shape=inputs[0].shape[1:], dtype=inputs[0].dtype), tf.TensorSpec(shape=inputs[1].shape[1:], dtype=inputs[1].dtype))
        return tf.map_fn(self.get_batch_wise, elems=inputs, fn_output_signature = output_sign)


class RandomGamma(Converter):

  def __init__(self, 
               probability=None, 
               gamma=0.4, 
               gain=0.75, 
               **kwargs):

      super(RandomGamma, self).__init__(**kwargs)

      self.probability = probability
      self.gamma = gamma
      self.gain = gain

  def adjust_gamma(self, inputs):

      gamma = (0, self.gamma)
      gamma = tf.random.uniform(shape=(), minval=gamma[0], maxval=gamma[1]) # Get values to use to translate x and y respectively

      gain = (self.gain/2, self.gain)
      gain = tf.random.uniform(shape=(), minval=gain[0], maxval=gain[1]) # Get values to use to translate x and y respectively
      img_gamma = tf.image.adjust_gamma(inputs[0], gamma=1+gamma, gain=gain)
      return img_gamma, inputs[1]
  
  def get_batch_wise(self, inputs):

      img, bboxes, rand = inputs[0], inputs[1], inputs[2]
      augment_value = tf.random.uniform(shape=tf.shape(rand))
      return tf.cond(tf.math.less(augment_value, rand), lambda: self.adjust_gamma((img, bboxes)), lambda: (img, bboxes))

  def call(self, inputs):
      
      output_sign = (tf.TensorSpec(shape=inputs[0].shape[1:], dtype=inputs[0].dtype), tf.TensorSpec(shape=inputs[1].shape[1:], dtype=inputs[1].dtype))
      return tf.map_fn(self.get_batch_wise, elems=inputs, fn_output_signature = output_sign)


class RandomRotate(Converter):

    def __init__(self, 
                 threshold = 0.3,
                 probability=None, 
                 rot=60,
                 **kwargs):
            
        super(RandomRotate, self).__init__(**kwargs)

        self.probability = probability
        self.angle = rot
        self.threshold = threshold

    def rot(self, inputs):

        img, bboxes_in = inputs[0], inputs[1]
        img_shape = tf.shape(img) # h, w, c

        rotate = (-np.abs(self.angle), np.abs(self.angle))
        degree_angle = tf.random.uniform(shape=(), minval=rotate[0], maxval=rotate[1]) # Get values to use to translate x and y respectively

        # Rotate Image
        R = tf.convert_to_tensor(([[ tf.math.cos( degree_angle*(np.pi/180) ), -tf.math.sin( degree_angle*(np.pi/180) )] , 
                                   [ tf.math.sin( degree_angle*(np.pi/180) ),  tf.math.cos( degree_angle*(np.pi/180) ) ]])) # Get rotation matrix
        img_Rotate = tfa.image.rotate(img, angles=degree_angle*(np.pi/180), interpolation='nearest', fill_mode='nearest')

        # Rotate Bounding Boxes
        bboxes = self.Un_Normalize(bboxes_in, img.shape)                                                                   # y1, x1, y2, x2
        bbx_extra = tf.transpose(tf.concat([[bboxes[:, 2], bboxes[:, 1], bboxes[:, 0], bboxes[:, 3]]], axis=-1)) # y2, x1, y1, x2
        bbx_extrA = tf.transpose(tf.concat([[bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 1]]], axis=-1)) # y2, x2, y1, x1
        bbx_extrB = tf.transpose(tf.concat([[bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]]], axis=-1)) # y1, x2, y2, x1
        coords = tf.concat([bboxes, bbx_extra, bbx_extrA, bbx_extrB], axis=1)

        img_center = tf.cast(img_shape[:2]/2, dtype=R.dtype) # h, w of image center or y/2, x/2
        coords_centered = tf.cast(tf.reshape(coords, (-1, 2)), dtype=R.dtype) - img_center # [::-1] [y, x], [y, x], [y, x], [y, x], ...
        coords_centered_T = tf.transpose(coords_centered) # [x, x, x, x, x, x, x, x, ...], [y, y, y, y, y, y, y, y, ...]
        coords_centered_T_R = R @ coords_centered_T # [2, 2] @ [2, n] -> # [x', x', x', x', x', x', x', x', ...], [y', y', y', y', y', y', y', y', ...]

        coords_R = tf.transpose(coords_centered_T_R) + img_center # [::-1] [y', x'], [y', x'], [y', x'], [y', x'], ...
        coords_R = tf.cast(tf.reshape(coords_R, tf.shape(coords)), dtype=R.dtype) # y1', x1', y2', x2', y2', x1', y1', x2', ...

        gather_y = [0, 2, 4, 6, 8, 10, 12, 14] ####### tf.range
        gather_x = [1, 3, 5, 7, 9, 11, 13, 15]
        new_y1 = K.min(tf.gather(coords_R, gather_y, axis=1), axis=1) # Get minimum y value 
        new_x1 = K.min(tf.gather(coords_R, gather_x, axis=1), axis=1) # Get minimum x value
        new_y2 = K.max(tf.gather(coords_R, gather_y, axis=1), axis=1) # Get maximum y value
        new_x2 = K.max(tf.gather(coords_R, gather_x, axis=1), axis=1) # Get maximum x value
        final_bbx_R = tf.concat([[new_y1, new_x1, new_y2, new_x2]], axis=0) # Make bbx with new coords
        final_bbx_R = tf.transpose(final_bbx_R)

        # Crop the bboxes if needed
        img_shape = tf.cast(img_shape, dtype=bboxes.dtype)
        final_bbx_R = tf.clip_by_value(final_bbx_R, 0, img_shape[0])
        final_bbx_R = self.Normalize(final_bbx_R, img_shape)

        # Postprocess Masking
        mask = self.mask(bboxes_in)
        final_bbx_R = final_bbx_R * mask
        final_bbx_R = self.remove_those_with_area_under_thresh(self.threshold, bboxes_in, final_bbx_R, img_shape)
        return img_Rotate, final_bbx_R
           
    def get_batch_wise(self, inputs):

        img, bboxes, rand = inputs[0], inputs[1], inputs[2]
        augment_value = tf.random.uniform(shape=tf.shape(rand))
        return tf.cond(tf.math.less(augment_value, rand), lambda: self.rot((img, bboxes)), lambda: (img, bboxes))

    def call(self, inputs):
        
        output_sign = (tf.TensorSpec(shape=inputs[0].shape[1:], dtype=inputs[0].dtype), tf.TensorSpec(shape=inputs[1].shape[1:], dtype=inputs[1].dtype))
        return tf.map_fn(self.get_batch_wise, elems=inputs, fn_output_signature = output_sign)


class AugProbability(layers.Layer):
   
    def __init__(self, **kwargs):
      
      super(AugProbability, self).__init__(**kwargs)
      self.probability = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="p")
      
    def call(self, inputs): # (1, 1) e.g. [[x]]
       
      self.probability.assign(inputs[0][0])

      return tf.repeat(self.probability, tf.shape(inputs)[0])
    

class Ada(tf.keras.Model):

    def __init__(self, 
                 img_size=(128, 128),
                 aug_functions=[],
                 initial_probability = 0.0,
                 switch = True,
                 **kwargs):
        
        super(Ada, self).__init__(**kwargs)
        
        self.probability = tf.Variable([[initial_probability]], name='ada_p')
        self.switch = switch
        self.augmenter = build_augmenter(aug_functions, img_size)
    
    def call(self, inputs, training=False):

        # Get inputs
        imgs, bboxes = inputs[0], inputs[1] # [bs, H, W, C], [bs, N, 5]
        bboxes, obj_class = tf.split(bboxes, [4, 1], axis=-1)

        # Repeat the probability for each image in the batch
        dim1 = tf.shape(self.probability)[0]
        bs = tf.shape(imgs)[0]
        probability = tf.repeat(self.probability, tf.cast(bs / dim1, tf.int32))

        # Perform augmentations
        if training and self.switch:
            aug_output = self.augmenter((imgs, bboxes, probability), training)
            return (aug_output[0], tf.concat([aug_output[1], obj_class], axis=-1))

        return inputs

    def update(self, loss): # Loss can be tf.constant of any shape

        # The more accurate the model, the probability of augmentations is higher.
        accuracy = K.mean(1. - loss)
        factor = (tf.math.exp(3. * accuracy) - 1) / (3. * tf.math.exp(accuracy) + 9.) # Arbitray function to map accuracy to 0-1 range.
        self.probability.assign( [[tf.clip_by_value(factor, 0.0, 1.0)]] )


def build_augmenter(aug_functions, img_size):

    if len(img_size) == 2:
        img_shape = img_size + (3,)

    if len(img_size) == 3:
        img_shape = img_size

    input_img = layers.Input(shape=img_shape) # (H, W, C)
    input_bboxes = layers.Input(shape=(None, 4))
    input_rand = layers.Input(shape=(1, ))

    p = AugProbability()(input_rand)
    x = (input_img, input_bboxes, p)
    for i, func in enumerate(aug_functions):
        out_img, out_bboxes = func(x)

        if i <= len(aug_functions) - 1:
          x = (out_img, out_bboxes, p)

    augment_model = tf.keras.Model([input_img, input_bboxes, input_rand], [out_img, out_bboxes], name='obj_det_data_augmentation_function')
    return augment_model
      