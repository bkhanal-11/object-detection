# ![Region-based Convolution Neural Network](https://arxiv.org/pdf/1311.2524v5.pdf)

As two-stage object detector, R-CNN follows following system overview.

![System Overview](assets/system-overview.png)

1. Takes an input image
2. Extracts around 2000 buttom-up region proposals
3. Computes features for each proposal using a large Convolutional Neural Network (CNN)
4. Classifes each region using class-specific linear SVMs (Support Vector Machine)

The RCNN (Region-based Convolutional Neural Networks) paper proposes a method for object detection that uses a combination of deep learning and traditional computer vision techniques. The approach is based on a region proposal algorithm that identifies potential object regions within an image, which are then classified using a convolutional neural network (CNN). The main contributions of the paper are:

1. Introducing a region proposal algorithm called Selective Search, which is used to generate potential object regions within an image.
    
2. Adapting a CNN architecture to perform object detection by training it on a large dataset of labeled object regions.
    
3. Combining the proposed region proposal algorithm and CNN architecture to create the RCNN object detection system.

The RCNN system achieved state-of-the-art results on the PASCAL VOC 2012 object detection dataset, with a mean average precision (mAP) of 53.3%.
Implementation Details

## Region Proposal Algorithm

The region proposal algorithm used in RCNN is called Selective Search. It works by combining image segmentation and hierarchical grouping to generate potential object regions. The algorithm works as follows:

1. Start by generating an initial set of small regions based on color and texture similarity.
    
2. Merge adjacent regions that have similar color and texture until a hierarchy of larger regions is obtained.

3. Use the hierarchy of regions to generate object proposals by recursively combining adjacent regions at different scales and thresholds.

## CNN Architecture

The CNN architecture used in RCNN is based on the AlexNet architecture, which consists of five convolutional layers, followed by three fully connected layers. The network is pre-trained on the ImageNet dataset, which contains over a million labeled images.

To adapt the pre-trained network to perform object detection, the last fully connected layer is replaced with a set of new layers that are trained on object regions labeled with the ground truth object class.

## Object Detection

To perform object detection using the RCNN system, the following steps are taken:

1. Use the Selective Search algorithm to generate potential object regions within an image.
   
2. For each potential object region, crop the image and resize it to a fixed size.
   
3. Pass the cropped image through the adapted CNN architecture to obtain a feature vector.
   
4. Use the feature vector to classify the object region using a linear SVM classifier trained on the labeled object regions.

5. Apply non-maximum suppression to eliminate redundant object proposals.

The RCNN system achieves state-of-the-art performance on several object detection datasets, including the PASCAL VOC and MS COCO datasets. However, it is relatively slow due to the need to run the CNN for each object proposal.
