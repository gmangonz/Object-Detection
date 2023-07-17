from tools.COCOdownload import download_and_extract_coco
from tools.COCOtoTFRecord import create_tf_records, create_tf_records_test
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', help="path to download and extract to", default=os.getcwd())
parser.add_argument('--tfRecords', '-t', help='Convert images to TFRecord files: 1 - yes, 0 - no', type=int, default=0)
parser.add_argument('--output', '-o', help="output path for TFRecord files", default=os.getcwd())
args = parser.parse_args()


if __name__ == '__main__':
  
    download_and_extract_coco(data_dir=args.path)

    if args.tfRecords == 1:
        create_tf_records(args.path, os.path.join(args.output, 'COCO_train.tfrecord'), split='train')
        create_tf_records(args.path, os.path.join(args.output, 'COCO_val.tfrecord'), split='val')
        create_tf_records_test(args.path, os.path.join(args.output, 'COCO_data_test.tfrecord'))
  