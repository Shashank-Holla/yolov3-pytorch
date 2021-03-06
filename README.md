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

    Here, `--images` flag defines the folder which contains image or multiple images. `--results` flag is the destination for the           processed images.


## Dependencies

- torch
- numpy
- pandas
- cv2

## Documentation

The following table contains functionalities and their corresponding module names.
| Functionality | Module Name |
|---------------|-------------|
| darknet       | Build yolov3 model. Load model's weights |
| detector      | Load and execute detection for images    |
| util          | Utility for functions such as bounding box, apply threshold and NMS etc.|


## Detection example

Shared below is a sample run results-

```
dog-cycle-car.jpg    predicted in  0.394 seconds
Objects Detected:    bicycle truck dog
----------------------------------------------------------
cat@dog.jpg          predicted in  0.097 seconds
Objects Detected:    cat dog
----------------------------------------------------------
giraffe.jpg          predicted in  0.098 seconds
Objects Detected:    zebra giraffe giraffe
----------------------------------------------------------
herd_of_horses.jpg   predicted in  0.094 seconds
Objects Detected:    horse horse horse horse
----------------------------------------------------------
streets-cars.jpg     predicted in  0.095 seconds
Objects Detected:    car car car car car car car car car car stop sign
----------------------------------------------------------
catdog2.jpg          predicted in  0.081 seconds
Objects Detected:    dog
----------------------------------------------------------
dogman.jpg           predicted in  0.081 seconds
Objects Detected:    person dog
----------------------------------------------------------
horse.jpg            predicted in  0.085 seconds
Objects Detected:    person person person person person traffic light horse horse horse horse horse horse
----------------------------------------------------------
dogman2.jpg          predicted in  0.073 seconds
Objects Detected:    person dog
----------------------------------------------------------
londonstreet.jpg     predicted in  0.078 seconds
Objects Detected:    person person person person person person car bus
----------------------------------------------------------
street.jpg           predicted in  0.085 seconds
Objects Detected:    person person person person car car car car car motorbike truck truck fire hydrant
----------------------------------------------------------
surf.jpg             predicted in  0.074 seconds
Objects Detected:    person surfboard
----------------------------------------------------------
SUMMARY
----------------------------------------------------------
Task                     : Time Taken (in seconds)

Reading addresses        : 0.001
Loading batch            : 11.009
Detection (12 images)    : 1.390
Output Processing        : 0.000
Drawing Boxes            : 2.797
Average time_per_img     : 1.306
----------------------------------------------------------
```



Object detection example-

![detection_image](https://github.com/Shashank-Holla/yolov3-pytorch/blob/master/results/results_streets-cars.jpg)

## Reference

This is based on blog series on yolov3- https://blog.paperspace.com/tag/series-yolo/
