import tensorflow as tf
from src.augmentations import Converter


class TransformBoxes(Converter):

    def __init__(self, 
                 grid_sizes = [8, 16, 32],
                 anchors = [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]],
                 image_shape = None,
                 **kwargs):

        super(TransformBoxes, self).__init__(**kwargs)
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

    def bboxes_to_grid(self, bboxes, grid_size, anchors, num_anchors):
        """
        Inputs:
            bboxes: shape - [BS, N, (y2, x2, y1, x1)]
            grid_size: int determining the grid size
            anchors: respective anchors to use [BS, W, H]
            num_anchors: number of anchors for the given grid size (aka num anchors for each scale)
        Output:
            anchor_targets - [BS, grid_size, grid_size, num_anchors, (y, x, h, w, p, obj_class)]
        """
        bboxes_shape      = tf.shape(bboxes)
        bs                = bboxes_shape[0]
        y_true_out        = tf.zeros((bs, grid_size, grid_size, num_anchors, 6)) # [BS, grid_size, grid_size, num_anchors, 6]
        bboxes, obj_class = tf.split(bboxes, [4, 1], axis=-1) # # [BS, N, 4], [BS, N, 1]
        
        mask       = self.mask(bboxes)[..., 0][..., None] # [BS, N, 1]
        bboxes     = self.yxyx_to_yxhw(bboxes)
        y, x, h, w = tf.split(bboxes, 4, axis=-1) # [BS, N, 1], [BS, N, 1], [BS, N, 1], [BS, N, 1] with values between 0 and 1

        # Get the cell bbox belongs to and the coordinates it would lie inside the cell
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
        probability = tf.ones_like(grid_y, dtype=bbox_coord_grid.dtype) # [BS, N, 1]
        updates     = tf.concat([bbox_coord_grid, probability, obj_class], axis=-1) * mask # [BS, N, 6]
        updates     = tf.reshape(updates, shape=[-1, 6]) # [BS * N, 6]
        
        # Filter out where we don't have bbox info
        final_updates = tf.boolean_mask(updates, tf.cast(tf.math.count_nonzero(updates, axis=-1), tf.bool)) # Remove those who don't have bboxes
        final_indices = tf.boolean_mask(indices, tf.cast(tf.math.count_nonzero(updates, axis=-1), tf.bool)) # Remove those who don't have bboxes

        # Get output
        return tf.tensor_scatter_nd_update(y_true_out, final_indices, final_updates) # y_true_out[batch][y][x][anchor] = (y, x, h, w, 1, class)
    
    def call(self, bboxes):

        a = self.bboxes_to_grid(bboxes, self.grid_sizes[0], self.anchors[0], num_anchors=3)
        b = self.bboxes_to_grid(bboxes, self.grid_sizes[1], self.anchors[1], num_anchors=3)
        c = self.bboxes_to_grid(bboxes, self.grid_sizes[2], self.anchors[2], num_anchors=3)
        return a, b, c


########################### CODE BELOW DOES THE SAME AS ABOVE BUT CAN ADD ASSIGN THE SECOND/THIRD BEST ANCHOR IN CASE THE OPTIMAL ANCHOR IS USED, HOWEVER IT IS ~20 TIMES SLOWER ###########################

input_signature = (tf.TensorSpec(shape=(None, None, None, 3, 6), dtype=tf.float32), # y_true_out
                   tf.TensorSpec(shape=(None, 3), dtype=tf.int32),                  # anchors_sorted_indxs
                   tf.TensorSpec(shape=(None, 3), dtype=tf.float32),                # iou_sorted
                   tf.TensorSpec(shape=(None, 3), dtype=tf.int32),                  # grid_yx
                   tf.TensorSpec(shape=(None, 6), dtype=tf.float32))                # data

@tf.function(input_signature=input_signature)
def make_output(y_true_out, anchors_sorted_indxs, iou_sorted, grid_yx, data):

    in_shape = tf.shape(anchors_sorted_indxs) # [Num_objects_in_whole_batch, 3]

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    
    idx     = 0
    indexes = indexes.write(idx, [-1, -1, -1, -1])
    updates = updates.write(idx, [-1, -1, -1, -1, -1, -1])
    
    for n in tf.range(in_shape[0]):

        # Get the respective values
        anchors_sorted     = anchors_sorted_indxs[n]
        anchors_iou_sorted = iou_sorted[n]  
        grid               = grid_yx[n]
        info               = data[n]
        values_seen        = indexes.stack()

        # Get optimal anchor
        optimal_anchor = anchors_sorted[0]

        # Set the new value to write
        new_value = tf.stack([grid[0], grid[1], grid[2], optimal_anchor])
        indexes   = indexes.write(idx, new_value)
        updates   = updates.write(idx, data[n])
        idx += 1

        # In this batch, and in the respective cell, determine if we have the optimal anchor taken
        element_wise_comparison = tf.reduce_all(tf.math.equal(values_seen, new_value), axis=-1, keepdims=True)

        # Optimal anchor has been taken
        if tf.reduce_any(element_wise_comparison):

            # "Erase" entry 
            idx -= 1

            # Get number of times he have used this cell in the grid
            num_instances_we_use_cell = tf.math.equal(values_seen[..., 0:-1], new_value[0:-1]) # If we are here, then this should always be atleast 1
            num_anchors_seen_in_cell  = tf.where(tf.reduce_all(num_instances_we_use_cell, axis=-1))
            num_anchors_seen_in_cell  = tf.shape(num_anchors_seen_in_cell)[0] # Value between 1 and 3

            # If we used at most 2 anchors for this cell, then grab the next one
            if num_anchors_seen_in_cell < 3:

                next_anchor = anchors_sorted[num_anchors_seen_in_cell] # Don't need to add +1 because the value is between 1 and 3
                anchor_iou  = anchors_iou_sorted[num_anchors_seen_in_cell]

                # Is the IOU meaningfull?
                if anchor_iou > 0.5:
                    # Write the anchor
                    indexes = indexes.write(idx, [grid[0], grid[1], grid[2], next_anchor])
                    updates = updates.write(idx, data[n])
                    idx += 1

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


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
        bboxes_shape      = tf.shape(bboxes)
        bs                = bboxes_shape[0]
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
        obj_class            = tf.gather(obj_class, importance, axis=1, batch_dims=-1) # [BS, N, 1]

        # Make indices
        indices   = tf.concat([grid_y, grid_x], axis=-1) # [BS, N, 2]
        indices   = tf.reshape(indices, shape=[-1, 2]) # [BS * N, 2]
        BS_grid   = tf.meshgrid(tf.range(bs), tf.range(bboxes_shape[1]))[0]
        BS_x_N    = tf.reshape(tf.transpose(BS_grid), [-1, 1]) # [BS * N, 1]
        indices   = tf.concat([BS_x_N, indices], axis=-1) # [BS * N, 3]

        # Make updates
        probability = tf.ones_like(grid_y, dtype=bbox_coord_grid.dtype) # [BS, N, 1]
        updates     = tf.concat([bbox_coord_grid, probability, obj_class], axis=-1) * mask # [BS, N, 6]

        # Reshape
        updates              = tf.reshape(updates, shape=[-1, 6]) # [BS * N, 6]
        anchors_sorted_indxs = tf.reshape(anchors_sorted_indxs, [-1, num_anchors]) # [BS * N, 3]
        iou_sorted           = tf.reshape(iou_sorted, [-1, num_anchors]) # [BS * N, 3]

        # Filter out where we don't have bbox info
        mask_out             = tf.cast(tf.math.count_nonzero(updates, axis=-1), tf.bool)
        anchors_sorted_indxs = tf.boolean_mask(anchors_sorted_indxs, mask_out)
        iou_sorted           = tf.boolean_mask(iou_sorted, mask_out)
        indices              = tf.boolean_mask(indices, mask_out)
        updates              = tf.boolean_mask(updates, mask_out)

        return make_output(y_true_out, anchors_sorted_indxs, iou_sorted, indices, updates)
    
    def call(self, bboxes):

        a = self.bboxes_to_grid(bboxes, self.grid_sizes[0], self.anchors[0], num_anchors=3)
        b = self.bboxes_to_grid(bboxes, self.grid_sizes[1], self.anchors[1], num_anchors=3)
        c = self.bboxes_to_grid(bboxes, self.grid_sizes[2], self.anchors[2], num_anchors=3)
        return a, b, c