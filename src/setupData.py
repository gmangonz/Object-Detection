import tensorflow as tf
from tools.read_tfRecord import read_tfrecord
import tensorflow_datasets as tfds
import os
from .utils import Un_Normalize, Normalize, yxyx_to_yxhw

def if_resize_with_path(image, bbox, image_shape):

    """
    Resize the image by keeping the aspect ratio
    
    Returns: 
        image - normalized resized image
        bbox - normalized bbox
    """

    img_h = tf.shape(image)[0]
    img_w = tf.shape(image)[1]
    new_h = image_shape[0]
    new_w = image_shape[1]
    scale = tf.constant([new_h, new_w, new_h, new_w], dtype=tf.float32)

    image = tf.image.resize_with_pad(image, image_shape[0], image_shape[1], method='nearest') / 255

    # Calcualte the size of the new image that will be placed at the center of the "canvas" of height new_h and width new_w
    h = tf.cast(img_h, tf.float32) * tf.cast(tf.math.minimum(new_w/img_w, new_h/img_h), tf.float32)
    w = tf.cast(img_w, tf.float32) * tf.cast(tf.math.minimum(new_w/img_w, new_h/img_h), tf.float32)
    ratio = tf.cast(tf.stack([h, w, h, w], axis=-1), dtype=bbox.dtype) / scale

    # Calculate how much padding is needed
    lower_h = (new_h-h)//2
    lower_w = (new_w-w)//2

    # Calculate the percentage that should be added to shift the bboxes correctly due to the padding
    add_to_bboxes_h = lower_h/h
    add_to_bboxes_w = lower_w/w

    bbox = (bbox + tf.cast(tf.stack([add_to_bboxes_h, add_to_bboxes_w, add_to_bboxes_h, add_to_bboxes_w], axis=-1), bbox.dtype)) * ratio
    return image, bbox

def _parse_features(example_proto, image_shape, features, resize_with_path=True):
    
    parsed_features = tf.io.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
    
    h     = tf.cast(parsed_features['image/height'], tf.int32)
    w     = tf.cast(parsed_features['image/width'], tf.int32)
    image = tf.reshape(image, shape=[h, w, 3])
    if not resize_with_path:
        image = tf.image.resize(image, size=image_shape, method='nearest') / 255
    
    num_objects = tf.shape(parsed_features['image/object/bbox/xmin'])[0]
    class_text  = tf.sparse.to_dense(parsed_features['image/object/class/text'])
    xmin        = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
    xmax        = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
    ymin        = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
    ymax        = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])
    bbox        = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    if resize_with_path:
        image, bbox = if_resize_with_path(image, bbox, image_shape)
    return image, bbox


def load_voc_dataset(args, voc_path, augment_func = None, split = 'train'):
    
    assert split in ['train', 'val']

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    print(f'Loading {split} dataset from {voc_path}...')

    features = read_tfrecord(voc_path)
    useful_features_keys = ['image/encoded', 'image/filename', 'image/format', 'image/height', 'image/width',
                            'image/object/bbox/xmax', 'image/object/bbox/xmin', 'image/object/bbox/ymax', 'image/object/bbox/ymin', 
                            'image/object/class/text', 'image/source_id']
    useful_features = {}
    for key in useful_features_keys:
        useful_features = dict(**useful_features, **{key: features[key]})
    print(f'Loaded useful features.')

    dataset = tf.data.TFRecordDataset(voc_path)
    dataset = dataset.shuffle(args.buffer_size)
    dataset = dataset.map(lambda x: _parse_features(x, image_shape = args.img_size, features = useful_features), num_parallel_calls=AUTOTUNE)
    dataset = dataset.padded_batch(args.batch_size, drop_remainder=True)
    if (augment_func != None) and (split == 'train'):
        dataset = dataset.map(lambda x, y: augment_func((x, y)), num_parallel_calls=tf.data.AUTOTUNE) # x - image, y - (bbox, bbox, bbox, bbox)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _parse_features_COCO(example_proto, image_shape, features, resize_with_path=True):
    
    parsed_features = tf.io.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
    
    h     = tf.cast(parsed_features['image/height'], tf.int32)
    w     = tf.cast(parsed_features['image/width'], tf.int32)
    image = tf.reshape(image, shape=[h, w, 3])
    if not resize_with_path:
        image = tf.image.resize(image, size=image_shape, method='nearest') / 255
    
    num_objects = tf.shape(parsed_features['image/object/bbox/xmin'])[0]
    class_label = tf.sparse.to_dense(parsed_features['image/object/class/label'])
    xmin        = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
    xmax        = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
    ymin        = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
    ymax        = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])
    bbox        = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    if resize_with_path:
        image, bbox = if_resize_with_path(image, bbox, image_shape)

    bbox = tf.concat([bbox, tf.cast(class_label[..., None], dtype=bbox.dtype)], axis=-1)
    return image, bbox


def load_coco_dataset(args, coco_path, augment_func = None, split = 'train'):
    
    assert split in ['train', 'val']

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    print(f'Loading {split} dataset from {coco_path}...')

    useful_features = {'image/height': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
                       'image/width': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
                       'image/filename': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                       'image/source_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                       'image/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                       'image/format': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                       'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                       'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                       'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                       'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                       'image/object/class/text': tf.io.VarLenFeature(dtype=tf.string),
                       'image/object/class/label': tf.io.VarLenFeature(dtype=tf.int64)}

    dataset = tf.data.TFRecordDataset(coco_path)
    dataset = dataset.shuffle(args.buffer_size)
    dataset = dataset.map(lambda x: _parse_features_COCO(x, image_shape = args.img_size, features = useful_features), num_parallel_calls=AUTOTUNE)
    dataset = dataset.padded_batch(args.batch_size, drop_remainder=True)
    if (augment_func != None) and (split == 'train'):
        dataset = dataset.map(lambda x, y: augment_func((x, y)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def setup_data(x, size):
  
  """
  Function passed in .map for tf.dataset.Dataset. Will setup the size for each image
  
  Input:
    x - sample from the dataset
    
  Returns:
    img - resized, normalized image to use
    
    label - tuple to use for the 4 outputs of the model
  
  """  

  img = x['image']
  img = tf.image.resize(img, size=size, method='nearest')
  img = img / 255
  
  bbx = x['objects']['bbox']
  
  yxyx = Un_Normalize(bbx, shape=img.shape)
  yxhw = yxyx_to_yxhw(yxyx)
  yxhw_normalized = Normalize(yxhw, shape=img.shape)

  return img, (yxhw_normalized, yxhw_normalized, yxhw_normalized, yxhw_normalized) # y_c, x_c, h, w all between 0 and 1; id