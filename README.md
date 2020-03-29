# yolov3-pytorch

## About
Implementation of yolo v3 Object detection using pytorch.

Yolo stands for You Only Look Once. Its an object detector that uses features learned by deep convolutional neural network to detect objects.

## How to use

### For single or multiple images

- Clone detector repository.

- As this code now contains only detector module, please download weights file from the official site-
`wget https://pjreddie.com/media/files/yolov3.weights` 

The following can be executed to run the model-
`python detector.py --images images/ --results results`

Here, `--images` flag defines the folder which contains image or multiple images. `--results` flag is the destination for the processed images.


## Dependencies

- torch
- numpy
- pandas
- cv2

## Documentation

The following table contains functionalities and their corresponding module names.
| Functionality | Module Name |
|---------------|-------------|
| darknet       | Building yolov3 model. Load model's weights |

## Reference

This is based on blog series on yolov3- https://blog.paperspace.com/tag/series-yolo/
