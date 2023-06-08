import tensorflow as tf

def calculate_iou(boxes1, boxes2):

    """Calculate the intersection over union of two set of boxes.
    boxes1: (tensor) bounding boxes, Shape: [num_objects,4]
    boxes2: (tensor) bounding boxes, Shape: [num_objects,4]

    Made my ChatGPT, which is ok as its actually kinda slow so will probably not use this.

    """
    # Expand dimensions for broadcasting
    boxes1 = tf.expand_dims(boxes1, axis=1) # [m, 1, 4]
    boxes2 = tf.expand_dims(boxes2, axis=0) # [1, n, 4]

    # Compute the intersection coordinates
    intersection_ymin = tf.maximum(boxes1[..., 0], boxes2[..., 0]) # [m, 1], [1, n] -> [max(m, n), min(m, n)]
    intersection_xmin = tf.maximum(boxes1[..., 1], boxes2[..., 1]) # [m, 1], [1, n] -> [max(m, n), min(m, n)]
    intersection_ymax = tf.minimum(boxes1[..., 2], boxes2[..., 2]) # [m, 1], [1, n] -> [max(m, n), min(m, n)]
    intersection_xmax = tf.minimum(boxes1[..., 3], boxes2[..., 3]) # [m, 1], [1, n] -> [max(m, n), min(m, n)]

    # Compute the intersection area
    intersection_area = tf.maximum(0.0, intersection_ymax - intersection_ymin) * tf.maximum(0.0, intersection_xmax - intersection_xmin) # [max(m, n), min(m, n)] 

    # Compute the box areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1]) # [m, 1]
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1]) # [1, n]

    # Compute the union area
    union_area = area1 + area2 - intersection_area # [max(m, n), min(m, n)] 

    # Compute the IoU
    iou = intersection_area / tf.maximum(union_area, 1e-6)

    return iou


def myIOU(boxesA, boxesB):
  
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


# batch_boxes1 = tf.constant([[10, 20, 50, 60], [30, 40, 70, 80], [15, 25, 55, 65]], dtype=tf.float32)
# batch_boxes2 = tf.constant([[20, 30, 60, 70], [35, 45, 53, 85]], dtype=tf.float32)

# time_start = time.time()
# iou = myIOU(batch_boxes1, batch_boxes2)
# time_end = time.time()

# print("Calculate IOU time: ", time_end - time_start)
# print("IoU:")
# print(iou.numpy())


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
  iou_scores = myIOU(boxes_pred, boxes_pred) # 256, 256
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


# boxes = tf.constant([[10., 20., 50., 60.], 
#                      [30., 40., 70., 80.], 
#                      [15., 25., 55., 65.], 
#                      [20., 30., 60., 70.]])

# selected_boxes = myNMS(boxes, threshold_1=0.65, threshold_2=0.3)

# print("Selected Boxes:")
# print(selected_boxes.numpy())