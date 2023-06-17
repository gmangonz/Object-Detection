import tensorflow as tf
from src.augmentations import Converter

class BboxesToAnchors(Converter):

    def __init__(self, 
                 grid_sizes = [13, 26, 52], # [13, 26, 52] 
                 anchors = [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]],
                 image_shape = None,
                 **kwargs):

        super(BboxesToAnchors, self).__init__(**kwargs)
        self.grid_sizes = grid_sizes
        self.anchors = anchors
        self.image_shape = image_shape

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

    @tf.function(experimental_relax_shapes=True)
    def make_output(self, y_true_out, anchors_sorted_indxs, iou_sorted, grid_yx, data):

        """
        Inputs:
            y_true_out: [BS, grid_size, grid_size, num_anchors, 6] initialized as 0's
            anchors_sorted_indxs: [BS, N, num_anchors] for each bbox, the anchor indices in order of IOU
            iou_sorted: [BS, N, num_anchors] sort the anchors for each box by IOU
            grid_yx: [BS, N, 2] y, x
            data: [BS, N, 6] y, x, h, w, p, c

        Output:
            y_true_out
        """

        out_shape = tf.shape(anchors_sorted_indxs) # [BS, N, num_anchors]

        indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        idx = 0

        indexes = indexes.write(idx, [-1, -1, -1, -1])
        updates = updates.write(idx, [-1, -1, -1, -1, -1, -1])
        
        for batch in tf.range(out_shape[0]):
            for n in tf.range(out_shape[1]):
                
                # Get the respective values
                grid = grid_yx[batch][n]
                info = data[batch][n]
                
                # If there is any coordinate info
                if tf.math.count_nonzero(info, dtype=tf.bool):
                    
                    values_seen = indexes.stack()

                    # Get anchors sorted by IOU for the respective bbox
                    anchors_sorted = anchors_sorted_indxs[batch][n]
                    anchors_iou_sorted = iou_sorted[batch][n]

                    # Get optimal anchor and its corresponding IOU
                    optimal_anchor = anchors_sorted[0]
                    optimal_iou = anchors_iou_sorted[0]

                    # Set the new value to write
                    gridy, gridx = tf.split(grid, 2)
                    new_value = tf.stack([batch, gridy[0], gridx[0], optimal_anchor])
                    indexes = indexes.write(idx, new_value)
                    updates = updates.write(idx, info)
                    idx += 1

                    # In this batch, and in the respective cell, determine if we have the optimal anchor taken
                    element_wise_comparison = tf.reduce_all(tf.math.equal(values_seen, new_value), axis=-1, keepdims=True)

                    # Optimal anchor has been taken
                    if tf.reduce_any(element_wise_comparison):

                        # "Erase" entry 
                        idx -= 1

                        # Get number of times he have used this cell in the grid
                        num_instances_we_use_cell = tf.math.equal(values_seen[..., 0:-1], new_value[0:-1]) # If we are here, then this should always be atleast 1
                        num_anchors_seen_in_cell = tf.where(tf.reduce_all(num_instances_we_use_cell, axis=-1))
                        num_anchors_seen_in_cell = tf.shape(num_anchors_seen_in_cell)[0] # Value between 1 and 3

                        # If we used at most 2 anchors for this cell, then grab the next one
                        if num_anchors_seen_in_cell < 3:

                            next_anchor = anchors_sorted[num_anchors_seen_in_cell] # Don't need to add +1 because the value is between 1 and 3
                            anchor_iou = anchors_iou_sorted[num_anchors_seen_in_cell]

                            # Is the IOU meaningfull?
                            if anchor_iou > 0.5:
                                # Write the anchor
                                new_value = tf.stack([batch, gridy[0], gridx[0], next_anchor])
                                indexes = indexes.write(idx, new_value)
                                updates = updates.write(idx, info)
                                idx += 1
        return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


    def bboxes_to_grid(self, bboxes, grid_size, anchors, num_anchors):
        """
        Inputs:
            bboxes: shape - [BS, N, (y, x, h, w, class_label)]
            grid_size: int determining the grid size
            anchors: respective anchors to use given the scale
            num_anchors: number of anchors for the given grid size (aka num anchors for each scale)
        Output:
            anchor_targets
        """
        bs                = tf.shape(bboxes)[0]
        y_true_out        = tf.zeros((bs, grid_size, grid_size, num_anchors, 6)) # [BS, grid_size, grid_size, num_anchors, 6]
        bboxes, obj_class = tf.split(bboxes, [4, 1], axis=-1) # # [BS, N, 4], [BS, N, 1]
        
        bboxes     = self.yxyx_to_yxhw(bboxes)
        mask       = self.mask(bboxes)[..., 0][..., None] # [BS, N, 1]
        y, x, h, w = tf.split(bboxes, 4, axis=-1) # [BS, N, 1], [BS, N, 1], [BS, N, 1], [BS, N, 1] with values between 0 and 1

        # Get the cell bbox belongs to and the coordinates it would lie inside the cell
        grid_y, grid_x  = tf.cast(grid_size * y, tf.int32), tf.cast(grid_size * x, tf.int32)  # [BS, N, 1], [BS, N, 1] where 1 is the respective index of the cell y, x fall into 
        y_inside_cell   = grid_size * y - tf.cast(grid_y, dtype=y.dtype)
        x_inside_cell   = grid_size * x - tf.cast(grid_x, dtype=x.dtype)
        bbox_coord_grid = tf.concat([y_inside_cell, x_inside_cell, h, w], axis=-1) # [BS, N, 4]

        # Get IOU between bboxes and anchors
        wh          = tf.concat([w, h], axis=-1) # get w, h as apparently that is what anchors are
        iou_anchors = self.iou_bbox_v_anchors(wh, anchors) # [BS, N, num_anchors] where (bs_i, N_j) and num_anchors_k is the IOU between wh[bs_i, N_j] and anchors[num_anchors_k]

        # Sort the IOU for each bbox and get the optimal value
        anchors_sorted_indxs = tf.argsort(iou_anchors, direction='DESCENDING', axis=-1) # [BS, N, num_anchors] get anchor indices in order of IOU
        iou_sorted           = tf.gather(iou_anchors, anchors_sorted_indxs, axis=-1, batch_dims=-1) # [BS, N, num_anchors] sort the anchors for each box
        iou_optimal_value    = iou_sorted[..., 0] # [BS, N] get the VALUE of the max IOU btwn bbox and optimal anchor
        importance           = tf.argsort(iou_optimal_value, direction='DESCENDING', axis=-1) # [BS, N]

        # Sort everything to be consistent - sort so bboxes with higher first IOU to ANY anchor box are prioritized
        anchors_sorted_indxs = tf.gather(anchors_sorted_indxs, importance, axis=1, batch_dims=-1) # [BS, N, num_anchors]
        iou_sorted           = tf.gather(iou_sorted, importance, axis=1, batch_dims=-1) # [BS, N, num_anchors]
        mask                 = tf.gather(mask, importance, axis=1, batch_dims=-1) # [BS, N, 1]
        bbox_coord_grid      = tf.gather(bbox_coord_grid, importance, axis=1, batch_dims=-1) # [BS, N, 4]
        grid_y               = tf.gather(grid_y, importance, axis=1, batch_dims=-1) # [BS, N, 1]
        grid_x               = tf.gather(grid_x, importance, axis=1, batch_dims=-1) # [BS, N, 1]

        # Make grid
        grid_yx     = tf.concat([grid_y, grid_x], axis=-1) # [BS, N, 2]

        # Make updates
        probability = tf.ones_like(grid_y, dtype=bbox_coord_grid.dtype) # [BS, N, 1]
        # obj_class   = tf.random.uniform(shape=grid_y.shape, minval=0, maxval=100, dtype=bbox_coord_grid.dtype) # # [BS, N, 1] <-------------------------------------------------------------------------
        updates     = tf.concat([bbox_coord_grid, probability, obj_class], axis=-1) # [BS, N, 6]

        y_out = self.make_output(y_true_out, anchors_sorted_indxs, iou_sorted, grid_yx * tf.cast(mask, tf.int32), updates * mask)
        return y_out
    
    def call(self, bboxes):

        a = self.bboxes_to_grid(bboxes, self.grid_sizes[0], self.anchors[0], num_anchors=3)
        b = self.bboxes_to_grid(bboxes, self.grid_sizes[1], self.anchors[1], num_anchors=3)
        c = self.bboxes_to_grid(bboxes, self.grid_sizes[2], self.anchors[2], num_anchors=3)
        return a, b, c