import tensorflow as tf
from model.augmentations import Converter


class TransformBoxes(Converter):

    def __init__(self, 
                 args,
                 grid_sizes = [8, 16, 32],
                 anchors = [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]],
                 **kwargs):

        super(TransformBoxes, self).__init__(**kwargs)
        self.num_scales = len(grid_sizes)
        self.grid_sizes = tf.convert_to_tensor(grid_sizes)
        self.anchors = tf.convert_to_tensor(anchors)
        self.num_anchors = tf.shape(self.anchors)[1]
        self.num_anchors_per_scale = tf.cast(tf.math.count_nonzero(self.anchors, axis=[0, -1]) / 2, dtype=tf.int32)
        self.args = args
        
        assert len(grid_sizes) == tf.shape(self.anchors)[0], 'Number of scales to use must match dimension 0 of anchors'

    def iou_bbox_v_anchors(self, boxes1, anchors):

        box1width, box1height = tf.split(boxes1, 2, axis=-1) # [bs, N, 1], [bs, N, 1]

        box2width, box2height = tf.split(anchors, 2, axis=-1) # [num_anchors, 1], [num_anchors, 1]
        box2width_fix = tf.transpose(box2width)[None, ...] # [num_anchors, 1] -> [1, num_anchors] -> [1, 1, num_anchors]
        box2height_fix = tf.transpose(box2height)[None, ...] # [num_anchors, 1] -> [1, num_anchors] -> [1, 1, num_anchors]

        min_width = tf.minimum(box1width, box2width_fix) # [bs, N, num_anchors]
        min_height = tf.minimum(box1height, box2height_fix) # [bs, N, num_anchors]

        intersection = min_width * min_height # [bs, N, num_anchors]

        box1_area = box1width * box1height # [bs, N, 1]
        box2_area = box2width * box2height # [num_anchors, 1]

        union = box1_area + tf.transpose(box2_area)[None, ...] - intersection
        return intersection / union # [bs, N, num_anchors] where bs_i, N_j, num_anchors_k is the IOU between boxes1[bs_i, N_j, ...] and anchors[num_anchors_k, ...]

    def bboxes_to_grid(self, bboxes, grid_size, anchors, num_anchors):
        """
        Inputs:
            bboxes: shape - [BS, N, (y2, x2, y1, x1, obj_class)]
            grid_size: int determining the grid size
            anchors: respective anchors to use [BS, W, H]
            num_anchors: number of anchors for the given grid size (aka num anchors for each scale)
        Output:
            anchor_targets - [BS, grid_size, grid_size, num_anchors, (y, x, h, w, p, obj_class)]
        """
        bboxes_shape      = tf.shape(bboxes)
        bs                = bboxes_shape[0]
        y_true_out        = tf.zeros((bs, grid_size, grid_size, num_anchors, tf.constant(6, dtype=tf.int32))) # [BS, grid_size, grid_size, num_anchors, 6]
        bboxes, obj_class = tf.split(bboxes, [4, 1], axis=-1) # # [BS, N, 4], [BS, N, 1]
        
        mask       = self.mask(bboxes)[..., 0][..., None] # [BS, N, 1]
        bboxes     = self.yxyx_to_yxhw(bboxes)
        y, x, h, w = tf.split(bboxes, 4, axis=-1) # [BS, N, 1], [BS, N, 1], [BS, N, 1], [BS, N, 1] with values between 0 and 1

        # Get the cell bbox belongs to and the coordinates it would lie inside the cell
        grid_size       = tf.cast(grid_size, bboxes.dtype)
        grid_y, grid_x  = tf.cast(grid_size * y, tf.int32), tf.cast(grid_size * x, tf.int32)  # [BS, N, 1], [BS, N, 1] where 1 is the respective index of the cell y, x fall into 
        y_inside_cell   = grid_size * y - tf.cast(grid_y, dtype=y.dtype)
        x_inside_cell   = grid_size * x - tf.cast(grid_x, dtype=x.dtype)
        bbox_coord_grid = tf.concat([y_inside_cell, x_inside_cell, h, w], axis=-1) # [BS, N, 4]

        # Get IOU between bboxes and anchors
        wh          = tf.concat([w, h], axis=-1) # get w, h as apparently that is what anchors are
        iou_anchors = self.iou_bbox_v_anchors(wh, anchors) * mask # [BS, N, num_anchors] where (bs_i, N_j) and num_anchors_k is the IOU between wh[bs_i, N_j] and anchors[num_anchors_k]

        # Sort the IOU for each bbox and get the optimal value
        anchors_sorted_indxs = tf.argsort(iou_anchors, direction='DESCENDING', axis=-1) # [BS, N, num_anchors] get anchor indices in order of IOU
        iou_sorted           = tf.gather(iou_anchors, anchors_sorted_indxs, axis=-1, batch_dims=-1) # [BS, N, num_anchors] sort the anchors for each box
        iou_optimal_value    = iou_sorted[..., 0] # [BS, N] get the VALUE of the max IOU btwn bbox and optimal anchor
        importance           = tf.argsort(iou_optimal_value, direction='DESCENDING', axis=1) # [BS, N]

        # Sort everything to be consistent - sort so bboxes with higher first IOU to ANY anchor box are prioritized
        anchors_sorted_indxs = tf.gather(anchors_sorted_indxs, importance, axis=1, batch_dims=-1) # [BS, N, num_anchors]
        iou_sorted           = tf.gather(iou_sorted, importance, axis=1, batch_dims=-1) # [BS, N, num_anchors]
        mask                 = tf.gather(mask, importance, axis=1, batch_dims=-1) # [BS, N, 1]
        bbox_coord_grid      = tf.gather(bbox_coord_grid, importance, axis=1, batch_dims=-1) # [BS, N, 4]
        grid_y               = tf.gather(grid_y, importance, axis=1, batch_dims=-1) # [BS, N, 1]
        grid_x               = tf.gather(grid_x, importance, axis=1, batch_dims=-1) # [BS, N, 1]
        optimal_anchors      = anchors_sorted_indxs[..., 0][..., None] # [BS, N, 1] get the INDEX of the optimal anchor 
        obj_class            = tf.gather(obj_class, importance, axis=1, batch_dims=-1) # [BS, N, 1]

        # Make indices
        indices   = tf.concat([grid_y, grid_x, optimal_anchors], axis=-1) # [BS, N, 3]
        indices   = tf.reshape(indices, shape=[-1, 3]) # [BS * N, 3]
        BS_grid   = tf.meshgrid(tf.range(bs), tf.range(bboxes_shape[1]))[0]
        BS_x_N    = tf.reshape(tf.transpose(BS_grid), [-1, 1])
        indices   = tf.concat([BS_x_N, indices], axis=-1) # [BS * N, 4]

        # Make updates
        confidence = tf.ones_like(grid_y, dtype=bbox_coord_grid.dtype) # [BS, N, 1]
        updates    = tf.concat([bbox_coord_grid, confidence, obj_class], axis=-1) * mask # [BS, N, 6]
        updates    = tf.reshape(updates, shape=[-1, 6]) # [BS * N, 6]
        
        # Filter out where we don't have bbox info
        final_updates = tf.boolean_mask(updates, tf.cast(tf.math.count_nonzero(updates, axis=-1), tf.bool)) # Remove those who don't have bboxes
        final_indices = tf.boolean_mask(indices, tf.cast(tf.math.count_nonzero(updates, axis=-1), tf.bool)) # Remove those who don't have bboxes

        # Get output
        transformed_bbox = tf.tensor_scatter_nd_update(y_true_out, final_indices, final_updates) # y_true_out[batch][y][x][anchor] = (y, x, h, w, 1, class)
        return transformed_bbox
        # return tf.RaggedTensor.from_tensor(transformed_bbox, ragged_rank=2)

    def call(self, bboxes):

        a = self.bboxes_to_grid(bboxes, self.grid_sizes[0], self.anchors[0], num_anchors=3) # [BS, grid_sizes[0], grid_sizes[0], 3, 6]
        b = self.bboxes_to_grid(bboxes, self.grid_sizes[1], self.anchors[1], num_anchors=3) # [BS, grid_sizes[1], grid_sizes[1], 3, 6]
        c = self.bboxes_to_grid(bboxes, self.grid_sizes[2], self.anchors[2], num_anchors=3) # [BS, grid_sizes[2], grid_sizes[2], 3, 6]
        return a, b, c