import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import os
import json

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_path, annotations):
    """Creates an Example proto for the given image.
    Args:
      image_path: Path to the image file.
      annotations: List of dicts in the format of {'id': id, 'bbox': bbox, ...}.
    Returns:
      tf_example: A tf.train.Example
    """
    # Get bytes of image
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        encoded_image_data = f.read()

    # Get image width and height
    image = Image.open(image_path)
    width, height = image.size

    # Get image file name and id
    filename = os.path.basename(image_path)
    image_id = int(os.path.splitext(filename)[0])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []

    # Get image info
    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        xmins.append(bbox[0] / width)
        ymins.append(bbox[1] / height)
        xmaxs.append((bbox[0] + bbox[2]) / width)
        ymaxs.append((bbox[1] + bbox[3]) / height)
        classes.append(category_id)
        classes_text.append(str(category_id).encode('utf8'))

    # Create tf.train.Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(str(image_id).encode('utf8')),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))

    return tf_example

def create_tf_records(data_dir, output_path, split='train'):
    
    assert split in ['train', 'val']

    image_dir = os.path.join(data_dir, f"{split}2017")
    annotations_file = os.path.join(data_dir, "annotations", f"instances_{split}2017.json")

    with tf.io.gfile.GFile(annotations_file, 'r') as f:
        annotations_data = json.load(f)

    images = annotations_data['images']
    annotations = annotations_data['annotations']

    writer = tf.io.TFRecordWriter(output_path)

    for image in tqdm(images):
        
        image_path = os.path.join(image_dir, image['file_name'])
        image_id = image['id']
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        tf_example = create_tf_example(image_path, image_annotations)
        writer.write(tf_example.SerializeToString())

    writer.close()

    print("TFRecord files created successfully!")


# data_directory = r"D:\DL-CV-ML Projects\Object Detection\Updated_Turion_Space\Test\Object Detection\data"
# output_tfrecord = r"D:\DL-CV-ML Projects\Object Detection\Updated_Turion_Space\Test\Object Detection\data\COCO_data_val.tfrecord"
# create_tf_records(data_directory, output_tfrecord, split='val')


def create_tf_example_test(image_path, info):

    # Get bytes from the image file
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        encoded_image_data = f.read()

    # Get image size
    image = Image.open(image_path)
    width, height = image.size

    # Get image info
    filename = info['file_name']
    image_id = info['id']
    height = info['height']
    width = info['width']

    # Create tf.train.Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(str(image_id).encode('utf8')),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature('jpeg'.encode('utf8'))
    }))

    return tf_example
    

def create_tf_records_test(data_dir, output_path):
    
    image_dir = os.path.join(data_dir, f"test2017")
    annotations_file = os.path.join(data_dir, "annotations", f"image_info_test2017.json")

    with tf.io.gfile.GFile(annotations_file, 'r') as f:
        annotations_data = json.load(f)

    images = annotations_data['images']

    writer = tf.io.TFRecordWriter(output_path)

    for image_info in tqdm(images):
        
        image_path = os.path.join(image_dir, image_info['file_name'])
        tf_example = create_tf_example_test(image_path, image_info)
        writer.write(tf_example.SerializeToString())

    writer.close()

    print("TFRecord files created successfully!")


# create_tf_records_test(data_directory, r"D:\DL-CV-ML Projects\Object Detection\Updated_Turion_Space\Test\Object Detection\data\COCO_data_test.tfrecord")