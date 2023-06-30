import tensorflow as tf


def Un_Normalize(bboxes_normalized, shape):

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
                       bboxes_normalized[:, 3] * shape[1]], axis=-1 )
    return bboxes


def Normalize(bboxes, shape):

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


def yxyx_to_yxhw(bboxes):
  
  """
  Input:
    bboxes: y1, x1, y2, x2
  
  Returns:
    yxhw: y, x, h, w - calcualted by [(y1, x1) + (y2, x2)] / 2 and [(y2, x2) - (y1, x1)]
  
  """

  return tf.concat( [ (bboxes[..., :2] + bboxes[..., 2:]) / 2.0, bboxes[..., 2:] - bboxes[..., :2] ], axis=-1 ) # y, x, h, w


def yxhw_to_yxyx(bboxes):

    """
    Input:
      bboxes: y_center, x_center, h, w 

    Returns:
      yxyx: bounding boxes containing y1, x1, y2, x2 
    
    """

    return tf.concat( [ bboxes[..., :2] - bboxes[..., 2:] / 2.0, bboxes[..., :2] + bboxes[..., 2:] / 2.0 ], axis=-1 )