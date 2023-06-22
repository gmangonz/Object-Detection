import tensorflow as tf
from src.utils import yxhw_to_yxyx

def bbox_iou(boxesA, boxesB):
  
  """
  bboxesA [BS, gy, gx, anchor, (yxyx)]
  bboxesB [BS, gy, gx, anchor, (yxyx])

  Returns: iou - (BS, gy, gx, anchor)
  """

  # Get coordinates
  yA_1, xA_1, yA_2, xA_2 = tf.split(boxesA, 4, axis=-1) # [BS, gy, gx, anchor, 1] x 4
  yB_1, xB_1, yB_2, xB_2 = tf.split(boxesB, 4, axis=-1) # [BS, gy, gx, anchor, 1] x 4

  # Find the coordinate of the overlapping boxes
  xI1 = tf.maximum(xA_1, tf.linalg.matrix_transpose(xB_1)) # [BS, gy, gx, anchor, 1], [BS, gy, gx, 1, anchor] -> [BS, gy, gx, anchor, anchor] 
  yI1 = tf.maximum(yA_1, tf.linalg.matrix_transpose(yB_1)) # [BS, gy, gx, anchor, 1], [BS, gy, gx, 1, anchor] -> [BS, gy, gx, anchor, anchor] 
  xI2 = tf.minimum(xA_2, tf.linalg.matrix_transpose(xB_2)) # [BS, gy, gx, anchor, 1], [BS, gy, gx, 1, anchor] -> [BS, gy, gx, anchor, anchor] 
  yI2 = tf.minimum(yA_2, tf.linalg.matrix_transpose(yB_2)) # [BS, gy, gx, anchor, 1], [BS, gy, gx, 1, anchor] -> [BS, gy, gx, anchor, anchor] 

  # Get area
  inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1) # [BS, gy, gx, anchor, anchor] 
  
  # Area of the original bounding boxes
  bboxesA_area = (xA_2 - xA_1 + 1) * (yA_2 - yA_1 + 1) # [BS, gy, gx, anchor, 1] 
  bboxesB_area = (xB_2 - xB_1 + 1) * (yB_2 - yB_1 + 1) # [BS, gy, gx, anchor, 1] 

  # Get overlap
  union = (bboxesA_area + tf.linalg.matrix_transpose(bboxesB_area)) - inter_area

  return tf.clip_by_value(inter_area / union, 0.0, 1.0) # [BS, gy, gx, anchor, anchor] 


def iou(boxesA, boxesB):
  
  """
  Compute IOU of a set of 2 bounding boxes. n == m is not always true, mostly when computing loss function
  
  Input:
    boxesA - first set of bounding boxes shape: (n, 4)
    
    boxesB - second set of bounding boxes shape: (m, 4)
    
  Output:
    
    IOU - (n, m) matrix where index (i, j) gives the IOU between n[i] and m[j]
  
  """
  
  # Get coordinates
  yA_1, xA_1, yA_2, xA_2 = tf.split(boxesA, 4, axis=1) # [n, 1]
  yB_1, xB_1, yB_2, xB_2 = tf.split(boxesB, 4, axis=1) # [m, 1]

  # Find the coordinate of the overlapping boxes
  xI1 = tf.maximum(xA_1, tf.transpose(xB_1)) # [n, 1], [1, m] -> [n, m]
  yI1 = tf.maximum(yA_1, tf.transpose(yB_1)) # [n, 1], [1, m] -> [n, m]
  xI2 = tf.minimum(xA_2, tf.transpose(xB_2)) # [n, 1], [1, m] -> [n, m]
  yI2 = tf.minimum(yA_2, tf.transpose(yB_2)) # [n, 1], [1, m] -> [n, m]

  # Get area
  inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1) # [n, m]
  
  # Area of the original bounding boxes
  bboxesA_area = (xA_2 - xA_1 + 1) * (yA_2 - yA_1 + 1) # [n, 1]
  bboxesB_area = (xB_2 - xB_1 + 1) * (yB_2 - yB_1 + 1) # [m, 1]

  # Get overlap
  union = (bboxesA_area + tf.transpose(bboxesB_area)) - inter_area

  return tf.clip_by_value(inter_area / union, 0.0, 1.0)


def NMS(boxes, scores, iou_threshold):
    
    """
    Compute NMS with scores
    boxes: [N, 4]
    scores: [N, 1] or [N,]
    iou_threshold = float between 0 and 1
    
    """
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=boxes.shape[0], iou_threshold=iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)
    
    return selected_boxes, selected_scores, selected_indices


def non_max_suppression(bbox, confs, obj_class_probs, num_classes, max_output_size=50, max_output_size_per_class=30, iou_threshold=0.5, confidence_threshold=0.5):

    """
    Inputs:
        bbox: [BS, gy, gx, NUM_ANCHORS, (y, x, h, w)]
        confs: [BS, gy, gx, NUM_ANCHORS, 1]
        obj_class_probs: [BS, gy, gx, NUM_ANCHORS, 1] or [BS, gy, gx, NUM_ANCHORS, NUM_CLASSES]
        max_output_size: int
        max_output_size_per_class: int
        iou_threshold: int
        confidence_threshold: int

    Outputs:
        boxes: [BS, max_detections, (y, x, y, x)] tensor containing the non-max suppressed boxes
        scores: [BS, max_detections] tensor containing the scores for the boxes.
        classes: [BS, max_detections] tensor containing the class for boxes.
        valid_detections:
    
    """

    bs = tf.shape(bbox)[0]
    bbox = yxhw_to_yxyx(bbox)
    bbox = tf.reshape(bbox, (bs, -1, 1, 4))

    obj_dim = tf.shape(obj_class_probs)[-1]
    obj_class_probs = tf.cond( tf.math.equal( obj_dim, num_classes ), lambda: obj_class_probs, lambda: tf.squeeze( tf.one_hot( tf.cast( obj_class_probs, tf.int32 ), num_classes, axis=-1 ) ) ) # [BS, gy, gx, NUM_ANCHORS, NUM_CLASSES]
    scores = confs * obj_class_probs # [BS, gy, gx, NUM_ANCHORS, NUM_CLASSES]
    scores = tf.reshape(scores, (bs, -1, tf.shape(scores)[-1])) # [BS, gy * gx * NUM_ANCHORS, NUM_CLASSES]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(boxes=bbox, # [batch_size, num_boxes, q, 4]
                                                                                     scores=scores, # [batch_size, num_boxes, num_classes] 
                                                                                     max_output_size_per_class=max_output_size_per_class,
                                                                                     max_total_size=max_output_size,
                                                                                     iou_threshold=iou_threshold,
                                                                                     score_threshold=confidence_threshold)
    return boxes, scores, classes, valid_detections


def decode_model_outputs(pred, anchors):

  """
  Inputs:
    pred: [BS, Grid Size Y, Grid Size X, NUM_ANCHORS, 4 + 1 + NUM_CLASSES] (y, x, h, w, probability, class_label)
    anchors: [BS, w, h]

  Outputs: 
    bbox: [BS, gy, gx, NUM_ANCHORS, (y, x, h, w)] - predicted global bbox -> used to calculate IOU
    confidence: [BS, gy, gx, NUM_ANCHORS, 1]
    obj_class: [BS, gy, gx, NUM_ANCHORS, NUM_CLASSES]
    pred_cell_box: [BS, gy, gx, NUM_ANCHORS, (y, x, h, w)] - predicted local bbox -> used for regression loss

  """

  grid_size = tf.shape(pred)[1:3]
  box_yx, box_hw, confidence, obj_class = tf.split(pred, (2, 2, 1, -1), axis=-1)

  box_yx        = tf.sigmoid(box_yx) # [BS, gy, gx, NUM_ANCHORS, 2]
  confidence    = tf.sigmoid(confidence) # [BS, gy, gx, NUM_ANCHORS, 1]
  obj_class     = tf.sigmoid(obj_class) # [BS, gy, gx, NUM_ANCHORS, NUM_CLASSES]
  pred_cell_box = tf.concat((box_yx, box_hw), axis=-1) # [BS, gy, gx, NUM_ANCHORS, 4]

  grid = tf.meshgrid(tf.range(grid_size[0], dtype=tf.float32), tf.range(grid_size[1], dtype=tf.float32))
  grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gy, gx, 1, 2]

  box_yx = (box_yx + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
  box_hw = tf.exp(box_hw) * anchors[..., ::-1]
  bbox   = tf.concat([box_yx, box_hw], axis=-1)

  return bbox, confidence, obj_class, pred_cell_box


