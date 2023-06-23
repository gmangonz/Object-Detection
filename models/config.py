import tensorflow as tf
from src.augmentations import RandomHorizontalFlip, RandomTranslate, RandomRotate, RandomZoom, RandomMirror, AddNoise, RandomGamma, GaussianBlur

class args:
    img_size = (256, 256)
    make_grid_size = False
    grid_size_ratio = 32
    num_scales = 3
    buffer_size = 64
    batch_size = 16
    epochs = 100
    NUM_CLASSES = 100
    aug_translate = 0.2
    aug_rot = 40
    aug_scale = 0.35
    aug_kernel_size = 5
    aug_sigma = 1
    ada_initial_probability = 0.0
    lambda_box = 5
    lambda_noobj = 10 # 0.5
    lambda_class = 1
    max_output_size = 50
    max_output_size_per_class = 30
    iou_threshold = 0.5
    confidence_threshold = 0.5
    ignore_thresh = 0.5
    optimizer = 'adam'
    monitor = 'loss'

config_opt = {"learning_rate": 1e-3, "beta_1": 0.15, "beta_2": 0.99, "epsilon": 1e-8}

grid_sizes = [8, 16, 32]
if args.make_grid_size:
    grid_sizes = []
    if args.num_scales != None:
        for i in range(args.num_scales):
            scale = (2 ** i * args.img_size[0]) // args.grid_size_ratio
            grid_sizes.append(scale)
    else:
        print('Please specify number of scales')

num_scales = len(grid_sizes)
norm_anchors = tf.constant([
                            [(116, 90), (156, 198), (373, 326)],
                            [(30, 61),   (62, 45),   (59, 119)], 
                            [(10, 13),   (16, 30),    (33, 23)]
                           ], tf.float32) / 416.
num_anchors_per_scale = tf.math.count_nonzero(norm_anchors, axis=[0, -1]) / 2
assert len(grid_sizes) == norm_anchors.shape[0]


voc_train_ds_path = r"D:\DL-CV-ML Projects\Turion_Space\Object Detection (OLD)\data\voc\voc2012_train.tfrecord"
voc_val_ds_path = r"D:\DL-CV-ML Projects\Turion_Space\Object Detection (OLD)\data\voc\voc2012_val.tfrecord"
coco_train_ds_path = r"D:\DL-CV-ML Projects\Turion_Space\Object Detection\data\COCO_data_train.tfrecord"
coco_val_ds_path = r"D:\DL-CV-ML Projects\Turion_Space\Object Detection\data\COCO_data_val.tfrecord"
img_path = r'D:\DL-CV-ML Projects\Turion_Space\Updated_Turion_Space\imgs\img.png'
save_model_path = r'D:\DL-CV-ML Projects\Turion_Space\Object Detection\Object Detection\model_weights\model.h5'

augmentations = [RandomHorizontalFlip(), 
                 RandomTranslate(translate=args.aug_translate), 
                 RandomRotate(rot=args.aug_rot), 
                 RandomZoom(scale=args.aug_scale), 
                 RandomMirror(), 
                 AddNoise(), 
                 RandomGamma(), 
                 GaussianBlur(kernel_size=args.aug_kernel_size, sigma=args.aug_sigma)]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]