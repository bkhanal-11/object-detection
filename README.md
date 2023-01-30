# Object Detection

Object detection is the field of computer vision that deals with the localization and classification of objects contained in an image or a video. Object detection comes down to drawing bounding boxes around detected objects which allow us to locate them in a given scene (or how they move through it).

Based on steps involved, object detection can be categorized into two:

1. Single-stage object detector
2. Two-stage object detector

Single-stage detector removes the RoI (Region of Interest) extraction process and directly classifies and regresses the candidate anchor boxes. YOLO (You Only Look Once) family is one of the example of single-stage detector. They invlove the use of single neural network trained end-to-end to take in an image as an input and predicts bounding boxes and class labels for each bounding box directly.

Two-stage detectors divide the object detection task into two stages: extract RoIs, then classify and regress the RoI. For example, R-CNN model family. R-CNN utilizes a selective search method to locate RoIs in the input images and uses a DCN (Deep Convolutional Neural Network)-based region-wise classifier to classify the RoIs independently. 