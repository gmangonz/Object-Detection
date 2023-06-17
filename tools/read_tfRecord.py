import tensorflow as tf

def get_tfrecordkeys(tfrecord_path):

    for rec in tf.data.TFRecordDataset([str(tfrecord_path)]):

        example_bytes = rec.numpy()
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        keys = []

        for key in example.features.feature:
            if key not in keys:
                keys.append(key)
    return keys


def read_tfrecord(tfrecord_path, return_values=False, coco=False):

    feature_dict = {}
    for rec in tf.data.TFRecordDataset([str(tfrecord_path)]):

        example_bytes = rec.numpy()
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        
        for key in example.features.feature:

            feature = example.features.feature[key]
            if key in ['image/object/bbox/xmin', 'image/object/bbox/ymin', 'image/object/bbox/xmax', 'image/object/bbox/ymax']:
                values = feature.float_list.value if return_values else tf.io.VarLenFeature(tf.float32)
            elif key == 'image/object/class/text':
                values = feature.bytes_list.value if return_values else tf.io.VarLenFeature(dtype=tf.string)
            elif key == 'image/object/class/label':
                values = feature.int64_list.value if return_values else tf.io.VarLenFeature(dtype=tf.int64)
            else: 
                if feature.HasField('bytes_list'):
                    values = feature.bytes_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.string)
                elif feature.HasField('float_list'):
                    values = feature.float_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.float32)
                elif feature.HasField('int64_list'):
                    values = feature.int64_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.int64)
                else:
                    values = feature.WhichOneof('kind')

            feature_dict[key] = values
            
        # Takes too long
        if coco:
            break
    return feature_dict