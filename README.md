# Object Detection

This repository contains my implementation of object detection from scratch in Tensorflow to better understand how the end to end process. I've tried to keep the code minimal and understandable, and comment on it as much as I can.

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
(in progress)

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

After running ```transform_bboxes.py```

  Output of large grid cells     |      Output of mid-sized grid cells        |
:-------------------------:|:------------------------:|
| <img src="images\image_grid_0.jpg" width=500px> | <img src="images\image_grid_1.jpg" width=500px> |

|      Output of small grid cells        |
:-------------------------:|
| <img src="images\image_grid_2.jpg" width=400> |

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
