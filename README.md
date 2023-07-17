# Object Detection

This repository contains my implementation of object detection from scratch in Tensorflow to better understand how the end to end process. I've tried to keep the code minimal and understandable, and comment on it as much as I can.

TODO:

- [x] Get end to end pipleline to work with dummy model
- [x] Build data augmentation
- [ ] Build model (waiting for GPU units in Google Colab)
- [ ] Train model (again GPU units)
- [ ] Display model architecture
- [ ] Get requirements.yaml file


# Requirements
1. Python 3.7
2. Tensorflow-gpu 2.6.0
3. Tensorflow-addons
4. Keras

Additional requirements listed in requirements.yaml file.

## Installation

Recommended to use [Anaconda](https://www.anaconda.com/)

```
conda env create -f requirements.yml
conda activate tf-gpu
```

# Usage
## Data Preparation

The data set used is the COCO 2017 dataset. Run the following to download the dataset.

## Training
(in progress)

## Perform detection
(in progress)

# How it works

This project has helped me understand the process behind object detectors which includes: data processing, model construction and output processing. Below I will explain the jist of these processes to enforce my understanding of the respective topics. 

## Running the data preprocessing

Below is sample image in the dataset.

<img src="images\image.jpg" width=500px> 

To initiate object detection, the data requires preprocessing to fit a specific input format for the model. Following the approach of YOLO (You Only Look Once), the data is processed at three different levels, where at each level the image is divided into distinct grid cell sizes. Now each cell is responsible for detecting 3 objects. Why 3 objects? Because we pre-define three anchor boxes that each cell uses to detect objects. 

```transform_bboxes.py``` is responsible for this data preprocessing which the algorithm can be described as follows:
```
For each grid size in [8, 16, 32]:
  Get the pre-defined anchor boxes for the grid size
  Make a matrix of zeros of size (BATCH SIZE, GRID SIZE Y, GRID SIZE X, 3, 6)
  For each image in the batch:
    For each bounding box in the image:
      Find the grid the bounding box belongs to. This is done in lines 64-68 in transform_bboxes.py
      Get the IOU between the bounding box and pre-defined anchors
      At the given batch, for the grid y and grid x, and for the anchor box with max IOU, assign the following coordinates (y, x, h, w, 1, CLASS) in the matrix previously created
```

I take advantage of broadcasting to avoid for loops and the results can be seen below. Blue boxes correspond to the bounding boxes and yellow boxes are the assigned pre-defined anchor boxes. The red dots correspond to the center of the bounding boxes used to locate the corresponding grid cell the object lies in. 

  Output of large grid cells     |      Output of mid-sized grid cells        |
:-------------------------:|:------------------------:|
| <img src="images\image_grid_0.jpg" width=500px> | <img src="images\image_grid_1.jpg" width=500px> |

|      Output of small grid cells        |
:-------------------------:|
| <img src="images\image_grid_2.jpg" width=400> |

Of course the question now becomes, what happens if 2 bounding boxes of different objects are in the same grid and whose max IOU is with the same anchor box? This is where ```bbox_preprocess.py``` comes in. It assigns the second best (and even third best) anchor box to the object in question. However, because this is very unlikely to occur, I stick with ```transform_bboxes.py``` and sort the IOU's between bounding box and anchors so that the anchor is assigned to the bounding box with the higher max IOU. 

## Data Augmentation

As a little bonus, I decided to implement adaptive data augmentation. This means that while the model trains, data augmentation will occur depending on the error produced by the model. For example, if the model is starting off and the error is high, the data augmentation will be minimal. However, once the error begins to decrease as the training continues, data augmentation will be occur at a higher frequency and different transformations will stack to add randomness. 

  Ada with high error (low accuracy)    |      Ada with lower error        |
:-------------------------:|:------------------------:|
| <img src="images\ada_high_error.jpg" width=600px> | <img src="images\ada_mid_error.jpg" width=600px> |

  Ada with even lower error      |      Ada with low error (high accuracy)       |
:-------------------------:|:------------------------:|
| <img src="images\ada_mid_low_error.jpg" width=600px> | <img src="images\ada_low_error.jpg" width=600px> |

## Model

Below displays the architecture of the model used ... (in progress)

## Outputs

(In progress)
