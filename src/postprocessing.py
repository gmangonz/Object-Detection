import tensorflow as tf

def bbox_iou(boxesA, boxesB):
  
  """
  bboxesA [BS, gy, gx, anchor, (yxyx)]
  bboxesB [BS, gy, gx, anchor, (yxyx])

  Returns: iou - (BS, gy, gx, anchor)
  """

  # Get coordinates
  yA_1, xA_1, yA_2, xA_2 = tf.split(boxesA, 4, axis=-1) # [BS, gy, gx, anchor, 1]
  yB_1, xB_1, yB_2, xB_2 = tf.split(boxesB, 4, axis=-1) # [BS, gy, gx, anchor, 1]

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

  return tf.clip_by_value(inter_area / union, 0.0, 1.0)


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


def myNMS(boxes_pred, threshold_1=0.7, threshold_2=0.4):
  
  """
  Compute NMS without scores
  
  Input:
    boxes_pred - bounding boxes predicted by model
    
    threshold_1 - if boxes overlap more than this, then "join"
    
    threshold_2 - if boxes overlap more less this, then they differ enough to be differen objects
    
  Output:
    unique_boxes - the set of unique bounding boxes
  
  """
  
  # Compute IOU
  iou_scores = iou(boxes_pred, boxes_pred) # 256, 256
  dim = iou_scores.shape[0]

  # Get the upper triangle of the matrix
  a = tf.range(dim)[tf.newaxis, :]
  b = tf.range(dim)[:, tf.newaxis]
  cond = tf.greater(a, b)
  upp_triangle = tf.cast(cond, tf.float32)
  iou_scores = iou_scores * upp_triangle # (1 - tf.eye(num_rows=dim)) * tf.cast(tfp.math.fill_triangular([1]*int(dim2), upper=True), dtype=iou_scores.dtype)
  
  # A common denominator will be found between the boxes that overlap
  max_iou_scores_horizontal = tf.math.reduce_max(iou_scores, axis=1)
  indices_of_boxes_overlap = tf.reshape(tf.where(max_iou_scores_horizontal > threshold_1), [-1])

  # Account for the boxes that don't overlap with any as well - the loners
  max_iou_scores_vertical = tf.math.reduce_max(iou_scores, axis=0)
  indices_of_boxes_unique = tf.reshape(tf.where(max_iou_scores_vertical < threshold_2), [-1])

  # Combine to get all the unique boxes
  indices_of_unique_boxes = tf.sets.union(indices_of_boxes_overlap[None, ...], indices_of_boxes_unique[None, ...]).values
  final_boxes = tf.gather(boxes_pred, indices_of_unique_boxes)

  return final_boxes


def decode_model_outputs(pred, anchors):

  """
  pred: [BS, Grid Size Y, Grid Size X, NUM_ANCHORS, 4 + 1 + NUM_CLASSES] (y, x, h, w, probability, class_label)
  anchors: [BS, w, h]
  
  """

  grid_size = tf.shape(pred)[1:3]
  box_yx, box_hw, confidence, obj_class = tf.split(pred, (2, 2, 1, 1), axis=-1)

  box_yx        = tf.sigmoid(box_yx) # [BS, gy, gx, NUM_ANCHORS, 2]
  confidence    = tf.sigmoid(confidence) # [BS, gy, gx, NUM_ANCHORS, 1]
  obj_class     = tf.sigmoid(obj_class) # [BS, gy, gx, NUM_ANCHORS, 1]
  pred_cell_box = tf.concat((box_yx, box_hw), axis=-1) # [BS, gy, gx, NUM_ANCHORS, 4]

  grid = tf.meshgrid(tf.range(grid_size[0], dtype=tf.float32), tf.range(grid_size[1], dtype=tf.float32))
  grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gy, gx, 1, 2]

  box_yx = (box_yx + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
  box_hw = tf.exp(box_hw) * anchors
  bbox   = tf.concat([box_yx, box_hw], axis=-1)

  return bbox, confidence, obj_class, pred_cell_box


