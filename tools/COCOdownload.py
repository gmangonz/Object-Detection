import os
import urllib.request
from zipfile import ZipFile
import json
import tensorflow as tf
import zipfile

def download_and_extract(url, data_dir, extract_path, split):

    # Download COCO dataset
    print(f"Downloading COCO {split} dataset...")
    download_path = os.path.join(data_dir, f"{split}2017.zip")
    urllib.request.urlretrieve(url, download_path)

    # Extract COCO dataset
    print(f"Extracting COCO {split} dataset...")
    with ZipFile(download_path, 'r') as zObject:
        zObject.extractall(path=extract_path)

def download_and_extract_coco(data_dir):

    # Paths to download
    train_coco_url = "http://images.cocodataset.org/zips/train2017.zip"
    val_coco_url = "http://images.cocodataset.org/zips/val2017.zip"
    test_coco_url = "http://images.cocodataset.org/zips/test2017.zip"
    train_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    extract_path = os.path.join(data_dir)


    download_and_extract(train_coco_url, data_dir, extract_path, "train")
    download_and_extract(val_coco_url, data_dir, extract_path, "val")
    download_and_extract(test_coco_url, data_dir, extract_path, "test")
    download_and_extract(train_annotations_url, data_dir, extract_path, "annotations_trainval")
    print("COCO dataset downloaded and extracted successfully!")


def prepCOCO(base_path, name):
  
  """
  Prepares the COCO dataset
  
  Inputs: 
    base_path - path to place zip file
    
    name - name of zip file
    
  """

  url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
  filename = os.path.join(base_path, f"{name}.zip")
  tf.keras.utils.get_file(filename, url)
  os.mkdir(os.path.join(base_path, name))
  with zipfile.ZipFile(filename, "r") as z_fp:
      z_fp.extractall(os.path.join(base_path, name))

# data_directory = r"D:\DL-CV-ML Projects\Object Detection\Updated_Turion_Space\Test\Object Detection\data"
# download_and_extract_coco(data_directory)