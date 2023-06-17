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

    def get_original_indices(self, indices):

        """
        Get the indices of original data

        Ex:
        indices = tf.constant([[9, 1, 2, 5],  #0 <---A*
                               [9, 2, 1, 3],  #1 <---B*
                               [9, 1, 2, 2],  #2      *
                               [9, 2, 1, 3],  #3 <---B
                               [9, 1, 2, 1],  #4 <---C*
                               [9, 1, 2, 1],  #5 <---C
                               [9, 1, 2, 5],  #6 <---A
                               [9, 2, 1, 3]]) #7 <---B
        
        Will give [0, 1, 2, 4] where there are *
        """

        # Make upper triangule
        Bs_x_N = tf.shape(indices)[0]
        a      = tf.range(Bs_x_N)[tf.newaxis, :]
        b      = tf.range(Bs_x_N)[:, tf.newaxis]
        uppT   = tf.cast(tf.greater(a, b), tf.float32)

        # Make broadcastable
        indices1 = indices[None, ...]
        indices2 = indices[:, None, :]

        # Get the original-copy pairs
        elem_wise_comparison       = tf.math.equal(indices1, indices2) # [1, BS*N, 4], [BS*N, 1, 4] -> [BS*N, BS*N, 4]
        comparison_btwn_each_item  = tf.reduce_all(elem_wise_comparison, axis=-1) # [BS*N, BS*N]
        get_where_duplicates_occur = tf.cast(comparison_btwn_each_item, tf.float32) * uppT # [BS*N, BS*N] Why multiply by upper traingle? Because the diagonal is always going to be True and to eliminate symmetry
        original_copy_pairs        = tf.cast(tf.where(get_where_duplicates_occur), tf.int32)
        tf.print(original_copy_pairs)

        # Get only the originals, remember that a copy can also be an original e.g [0, 1] [1, 2] this means 0 == 1 == 2, but we only need 0
        unique_originals = tf.unique(original_copy_pairs[..., 0])[0]
        unique_copies    = tf.unique(original_copy_pairs[..., 1])[0]
        original         = tf.sets.difference(unique_originals[None, ...], unique_copies[None, ...]).values

        # From the original data, get the indices of non duplicates
        originals = tf.gather(get_where_duplicates_occur, original)
        copies    = tf.cast(tf.where(originals)[..., 1], tf.int32)
        all_idxs  = tf.range(Bs_x_N)

        # Get output
        non_duplicates = tf.sets.difference(all_idxs[None, ...], copies[None, ...], aminusb=True).values
        return non_duplicates


    def bboxes_to_grid(self, bboxes, grid_size, anchors, num_anchors, scale):
        """
        Inputs:
            bboxes: shape - [BS, N, 4] - y, x, h, w
            grid_size: int determining the grid size
            anchors: respective anchors to use
            num_anchors: number of anchors for the given grid size (aka num anchors for each scale)
            scale: respective scale for the given grid size
        Output:
            anchor_targets
        """
        bs         = tf.shape(bboxes)[0]
        y_true_out = tf.zeros((bs, grid_size, grid_size, num_anchors, 6)) # [BS, grid_size, grid_size, num_anchors, 6]
        bboxes     = self.yxyx_to_yxhw(bboxes)
        mask       = self.mask(bboxes)[..., 0][..., None] # [BS, N, 1]
        y, x, h, w = tf.split(bboxes, 4, axis=-1) # [BS, N, 1], [BS, N, 1], [BS, N, 1], [BS, N, 1] with values between 0 and 1

        # Get the cell bbox belongs to and the coordinates it would lie inside the cell
        grid_y, grid_x  = tf.cast(scale * y, tf.int32), tf.cast(scale * x, tf.int32)  # [BS, N, 1], [BS, N, 1] where 1 is the respective index of the cell y, x fall into 
        y_inside_cell   = scale * y - tf.cast(grid_y, dtype=y.dtype)
        x_inside_cell   = scale * x - tf.cast(grid_x, dtype=x.dtype)
        bbox_coord_grid = tf.concat([y_inside_cell, x_inside_cell, h, w], axis=-1) # [BS, N, 4]

        # Get IOU between bboxes and anchors
        wh                   = tf.concat([w, h], axis=-1) # get w, h as apparently that is what anchors are
        iou_anchors          = self.iou_bbox_v_anchors(wh, anchors) # [BS, N, num_anchors] where (bs_i, N_j) and num_anchors_k is the IOU between wh[bs_i, N_j] and anchors[num_anchors_k]
        anchors_sorted_indxs = tf.argsort(iou_anchors, direction='DESCENDING', axis=-1) # [BS, N, num_anchors] get anchor indices in order of IOU

        # Sort the anchors for each box and sort the bboxes so that the one with larger max IOU to ANY anchor box goes first
        iou_sorted        = tf.gather(iou_anchors, anchors_sorted_indxs, axis=-1, batch_dims=-1) # [BS, N, num_anchors] sort the anchors for each box
        iou_optimal_value = iou_sorted[..., 0] # [BS, N] get the VALUE of the max IOU btwn bbox and optimal anchor
        importance        = tf.argsort(iou_optimal_value, direction='DESCENDING', axis=-1) # [BS, N]

        # Sort optimal anchors, if for a cell there 1> anchors needed, whenever tf.tensor_scatter_nd_update updates values at a given index, it keeps the last one a.k.a. the one with higher IOU
        optimal_anchors = anchors_sorted_indxs[..., 0][..., None] # [BS, N, 1] get the INDEX of the optimal anchor 
        optimal_anchors = tf.gather(optimal_anchors, importance, axis=1, batch_dims=-1) # [BS, N, 1] sort so bboxes with higher first IOU to ANY anchor box go first
        anchors_sorted_indxs = tf.gather(anchors_sorted_indxs, importance, axis=1, batch_dims=-1) # [BS, N, num_anchors] sort so bboxes with higher first IOU to ANY anchor box go first

        # Sort following to keep consistency
        grid_y          = tf.gather(grid_y, importance, axis=1, batch_dims=-1) # [BS, N, 1]
        grid_x          = tf.gather(grid_x, importance, axis=1, batch_dims=-1) # [BS, N, 1]
        mask            = tf.gather(mask, importance, axis=1, batch_dims=-1) # [BS, N, 1]
        bbox_coord_grid = tf.gather(bbox_coord_grid, importance, axis=1, batch_dims=-1) # [BS, N, 4]
        iou_optimal_val = tf.gather(iou_optimal_value[..., None], importance, axis=1, batch_dims=-1) # REALLY ONLY FOR DEBUGGUGING PURPOSES

        # Make indices
        indices   = tf.concat([grid_y, grid_x, optimal_anchors], axis=-1) * tf.cast(mask, tf.int32) # [BS, N, 3]
        indices   = tf.reshape(indices, shape=[-1, 3]) # [BS * N, 3]
        BS_x_N    = tf.range(tf.shape(indices)[0])[..., None] # [BS * N, 1]
        batch_idx = tf.math.mod(BS_x_N, bs) # [BS * N, 1] of values 0 to BS-1. This is to keep track of the batch index after tf.reshape 
        indices   = tf.cast(tf.concat([batch_idx, indices], axis=-1), tf.int32) # [BS * N, 4]
        indices   = indices * tf.reshape(tf.cast(mask, tf.int32), [-1, 1]) # [BS * N, 4]

        # Make updates
        probability = tf.ones_like(optimal_anchors, dtype=bbox_coord_grid.dtype) # [BS, N, 1]
        obj_class   = tf.random.uniform(shape=optimal_anchors.shape, minval=0, maxval=100, dtype=bbox_coord_grid.dtype) # # [BS, N, 1] <-------------------------------------------------------------------------
        updates     = tf.concat([bbox_coord_grid, probability, obj_class], axis=-1) * mask # [BS, N, 6]
        updates     = tf.reshape(updates, shape=[-1, 6]) # [BS * N, 6]

        # Filter out [0, 0, 0, 0]'s from indices; however in the case that in batch 0, gridy 0, gridx 0, anchor 0 is needed for a bbox, we filter those that don't have bboxes
        final_updates = tf.boolean_mask(updates, tf.cast(tf.math.count_nonzero(updates, axis=-1), tf.bool)) # Remove those who don't have bboxes
        final_indices = tf.boolean_mask(indices, tf.cast(tf.math.count_nonzero(updates, axis=-1), tf.bool)) # Remove those who don't have bboxes

        # Get non duplicates
        true_indices  = self.get_original_indices(final_indices)
        tf.print(true_indices)
        
        final_indices = tf.gather(final_indices, true_indices)
        final_updates = tf.gather(final_updates, true_indices)

        ##################################################### TODO #####################################################
        # FOR DUPLICATES GET THE NEXT ANCHOR IF IOU > 0.5
        anchors_sorted_indxs_next  = tf.roll(anchors_sorted_indxs, -1, axis=-1)[..., 0][..., None]
        ################################################################################################################    

        # Get output
        anchor_targets = tf.tensor_scatter_nd_update(y_true_out, final_indices, final_updates) # y_true_out[batch][y][x][anchor] = (y, x, h, w, 1, class)
        return anchor_targets
    
    def call(self, bboxes):

        a = self.bboxes_to_grid(bboxes, self.grid_sizes[0], self.anchors[0], num_anchors=3, scale=self.grid_sizes[0])
        b = self.bboxes_to_grid(bboxes, self.grid_sizes[1], self.anchors[1], num_anchors=3, scale=self.grid_sizes[1])
        c = self.bboxes_to_grid(bboxes, self.grid_sizes[2], self.anchors[2], num_anchors=3, scale=self.grid_sizes[2])
        return a, b, c