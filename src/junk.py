## JUNK CODE I DON'T WANT TO THROW AWAY YET


# y_true1 = grid[0]
# y_pred1 = tf.random.uniform(tf.shape(y_true1) + (0, 0, 0, 0, 99))
# y_true2 = grid[1]
# y_pred2 = tf.random.uniform(tf.shape(y_true2) + (0, 0, 0, 0, 99))
# y_true3 = grid[2]
# y_pred3 = tf.random.uniform(tf.shape(y_true3) + (0, 0, 0, 0, 99))


# bbox_true1, conf1, obj_class1 = tf.split(y_true1, [4, 1, 1], axis=-1)
# obj_class1 = tf.squeeze(tf.one_hot(tf.cast(obj_class1, tf.int32), 100, axis=-1))
# obj_class1 = tf.math.log(obj_class1 / (1 - obj_class1))
# conf1 = tf.math.log(conf1 / (1 - conf1))
# y_pred1 = tf.concat([bbox_true1, conf1, obj_class1], axis=-1)
# small_noise1 = tf.random.normal(tf.shape(y_pred1), mean=0, stddev=.1)
# y_pred1 = y_pred1+small_noise1

# bbox_true2, conf2, obj_class2 = tf.split(y_true2, [4, 1, 1], axis=-1)
# obj_class2 = tf.squeeze(tf.one_hot(tf.cast(obj_class2, tf.int32), 100, axis=-1))
# obj_class2 = tf.math.log(obj_class2 / (1 - obj_class2))
# conf2 = tf.math.log(conf2 / (1 - conf2))
# y_pred2 = tf.concat([bbox_true2, conf2, obj_class2], axis=-1)
# small_noise2 = tf.random.normal(tf.shape(y_pred2), mean=0, stddev=.1)
# y_pred2 = y_pred2+small_noise2

# bbox_true3, conf3, obj_class3 = tf.split(y_true3, [4, 1, 1], axis=-1)
# obj_class3 = tf.squeeze(tf.one_hot(tf.cast(obj_class3, tf.int32), 100, axis=-1))
# obj_class3 = tf.math.log(obj_class3 / (1 - obj_class3))
# conf3 = tf.math.log(conf3 / (1 - conf3))
# y_pred3 = tf.concat([bbox_true3, conf3, obj_class3], axis=-1)
# small_noise3 = tf.random.normal(tf.shape(y_pred3), mean=0, stddev=.1)
# y_pred3 = y_pred3+small_noise3


# print(K.mean(loss1(y_true1, y_pred1)))
# print('--------------------------------')
# print(K.mean(loss2(y_true2, y_pred2)))
# print('--------------------------------')
# print(K.mean(loss3(y_true3, y_pred3)))

    #   count_pos_values = tf.math.count_nonzero(bboxes, axis=-1, dtype=tf.float32) # Count num of nonzeros in the rows
    #   rows_to_keep = tf.clip_by_value(count_pos_values, 0, 1)[None, ...] # Any row that had a nonzero value will be kept

    #   perm_T = tf.roll(tf.range(0, tf.rank(bboxes)), -1, axis=0) # Get the permutation
    #   col_vector = tf.transpose(rows_to_keep, perm_T) # Transform the rows_to_keep into a column vector 
      
    #   mask = tf.repeat(col_vector, 4, axis=-1) # Repeat the column vector 4 times to match bboxes
    #   return mask


# bvdmfs =0
# for imgs in kvsdk[1]:
#     bvdmfs+=tf.math.count_nonzero(tf.math.count_nonzero(imgs, axis=-1, dtype=tf.bool))
# bvdmfs

# b = make_output(tf.random.uniform((12, 8, 8, 3, 6)), tf.random.uniform((12, 38, 3), maxval=10, dtype=tf.int32), tf.random.uniform((12, 38, 3)), tf.random.uniform((12, 38, 2), maxval=8, dtype=tf.int32), tf.random.uniform((12, 38, 6)))

    #     # # Get non duplicates
    #     # true_indices  = self.get_original_indices(final_indices)
    #     # final_indices = tf.gather(final_indices, true_indices)
    #     # final_updates = tf.gather(final_updates, true_indices)  

    # def get_original_indices(self, indices):

    #     """
    #     Get the indices of original data

    #     Ex:
    #     indices = tf.constant([[9, 1, 2, 5],  #0 <---A*
    #                            [9, 2, 1, 3],  #1 <---B*
    #                            [9, 1, 2, 2],  #2      *
    #                            [9, 2, 1, 3],  #3 <---B
    #                            [9, 1, 2, 1],  #4 <---C*
    #                            [9, 1, 2, 1],  #5 <---C
    #                            [9, 1, 2, 5],  #6 <---A
    #                            [9, 2, 1, 3]]) #7 <---B
        
    #     Will give [0, 1, 2, 4] where there are *
    #     """

    #     # Make upper triangule
    #     Bs_x_N = tf.shape(indices)[0]
    #     a      = tf.range(Bs_x_N)[tf.newaxis, :]
    #     b      = tf.range(Bs_x_N)[:, tf.newaxis]
    #     uppT   = tf.cast(tf.greater(a, b), tf.float32)

    #     # Make broadcastable
    #     indices1 = indices[None, ...]
    #     indices2 = indices[:, None, :]

    #     # Get the original-copy pairs
    #     elem_wise_comparison       = tf.math.equal(indices1, indices2) # [1, BS*N, 4], [BS*N, 1, 4] -> [BS*N, BS*N, 4]
    #     comparison_btwn_each_item  = tf.reduce_all(elem_wise_comparison, axis=-1) # [BS*N, BS*N]
    #     get_where_duplicates_occur = tf.cast(comparison_btwn_each_item, tf.float32) * uppT # [BS*N, BS*N] Why multiply by upper traingle? Because the diagonal is always going to be True and to eliminate symmetry
    #     original_copy_pairs        = tf.cast(tf.where(get_where_duplicates_occur), tf.int32)

    #     # Get only the originals, remember that a copy can also be an original e.g [0, 1] [1, 2] this means 0 == 1 == 2, but we only need 0
    #     unique_originals = tf.unique(original_copy_pairs[..., 0])[0]
    #     unique_copies    = tf.unique(original_copy_pairs[..., 1])[0]
    #     original         = tf.sets.difference(unique_originals[None, ...], unique_copies[None, ...]).values

    #     # From the original data, get the indices of non duplicates
    #     originals = tf.gather(get_where_duplicates_occur, original)
    #     copies    = tf.cast(tf.where(originals)[..., 1], tf.int32)
    #     all_idxs  = tf.range(Bs_x_N)

    #     # Get output
    #     non_duplicates = tf.sets.difference(all_idxs[None, ...], copies[None, ...], aminusb=True).values
    #     return non_duplicates


# vsddsv = tf.constant([[4, 5, 6, 7], 
#                       [0, 1, 2, 0],
#                       [0, 1, 2, 1]])
# ppppp = tf.constant([0, 1, 2])

# num_instances_we_use_grid = tf.math.equal(vsddsv[1:, 0:-1], ppppp)
# num_anchors_seen_in_grid = tf.where(tf.reduce_all(num_instances_we_use_grid, axis=-1))
# tf.shape(num_anchors_seen_in_grid)[0]
# batch_boxes1 = tf.random.uniform(shape=(1, 2, 4))
# print(batch_boxes1)
# batch_boxes2 = anchors[:3]
# batch_boxes1[..., 2:][..., ::-1]
# print("out ---------------------------------------------")
# out = iou_width_height(batch_boxes1[..., 2:][..., ::-1], batch_boxes2)
# print(out)
# print("sortedd ---------------------------------------------")
# sortedd = tf.argsort(out, direction='DESCENDING', axis=-1)
# print(sortedd)
# print("iou_sorted ---------------------------------------------")
# iou_sorted = tf.gather(out, sortedd, axis=-1, batch_dims=-1)
# print(iou_sorted)
# print("opt ---------------------------------------------")
# opt = tf.argmax(out, axis=-1)[..., None]
# print(opt)
# print("val_of_optimal_iou ---------------------------------------------")
# val_of_optimal_iou = iou_sorted[..., 0]
# print(val_of_optimal_iou)
# print("important_sort ---------------------------------------------")
# important_sort = tf.argsort(val_of_optimal_iou, direction='DESCENDING', axis=-1)
# print(important_sort)
# print("sor ---------------------------------------------")
# sor = tf.gather(iou_sorted, important_sort, axis=1, batch_dims=-1)
# print(sor)
# print("sorted_opt ---------------------------------------------")
# sorted_opt = tf.gather(opt, important_sort, axis=1, batch_dims=-1)
# print(sorted_opt)
# i = -3
# img_to_resize = kvsdk[0][i]
# its_bboxes = kvsdk[1][i]
# visualize_boxes(img_to_resize,its_bboxes*tf.constant([256, 256, 256, 256], dtype=tf.float32), figsize=(5, 5), linewidth=1, color=[0, 0, 1])

# new_h = 239
# new_w = 312
# scale = tf.constant([new_h, new_w, new_h, new_w], dtype=tf.float32)
# img_to_resize_out = tf.image.resize_with_pad(img_to_resize, target_height = new_h, target_width = new_w)


# img_h = img_to_resize.shape[0]
# img_w = img_to_resize.shape[1]

# # Calcualte the size of the new image that will be placed at the center of the "canvas" of height new_h and width new_w
# h = img_h * tf.math.minimum(new_w/img_w, new_h/img_h)
# w = img_w * tf.math.minimum(new_w/img_w, new_h/img_h)
# ratio = tf.cast(tf.concat([h, w, h, w], axis=0), dtype=its_bboxes.dtype) / scale

# # Calculate how much padding is needed
# lower_h = (new_h-h)//2
# upper_h = (new_h-h)//2 + h
# lower_w = (new_w-w)//2
# upper_w = (new_w-w)//2 + w

# # Calculate the percentage that should be added to shift the bboxes correctly due to the padding
# add_to_bboxes_h = lower_h/h
# add_to_bboxes_w = lower_w/w

# new_bboxes = (its_bboxes + tf.cast(tf.concat([add_to_bboxes_h, add_to_bboxes_w, add_to_bboxes_h, add_to_bboxes_w], axis=0), its_bboxes.dtype)) * ratio

# visualize_boxes(img_to_resize_out, new_bboxes*scale, figsize=(5, 5), linewidth=1, color=[0, 0, 1])