import os
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

def IoU(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1_x1,box2_x1)
    yi1 = max(box1_y1,box2_y1)
    xi2 = min(box1_x2,box2_x2)
    yi2 = min(box1_y2,box2_y2)
    inter_width = xi2 - xi1
    inter_height =  yi2 - yi1
    inter_area = max(inter_width, 0) * max(inter_height, 0)
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_y2 - box1_y1)*(box1_x2 - box1_x1)
    box2_area = (box2_y2 - box2_y1)*(box2_x2 - box2_x1)
    union_area = (box1_area + box2_area) - inter_area
    
    # compute the IoU
    iou = inter_area / (union_area + 1e-7)
    
    return iou

class FineTuneDataset(Dataset):
    def __init__(self, image_path, annot_path):
        self.image_path = image_path
        self.annot_path = annot_path
        self.ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.train_images = []
        self.train_labels = []
        self._load_data()

    def __getitem__(self, idx):
        image = self.train_images[idx] / 255.0
        label = self.train_labels[idx]
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image).float(), torch.tensor(label).float()

    def __len__(self):
        return len(self.train_labels)

    def _load_data(self):
        for e, i in tqdm(enumerate(os.listdir(self.annot_path))):
            filename = i.split(".")[0] + ".jpg"
            image = cv.imread(os.path.join(self.image_path, filename))
            df = pd.read_csv(os.path.join(self.annot_path, i))
            gtvalues = []
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                gtvalues.append([x1, x2, y1, y2])

            self.ss.setBaseImage(image)
            self.ss.switchToSelectiveSearchFast()
            ssresults = self.ss.process()

            imout = image.copy()   
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
            for e, result in enumerate(ssresults):
                if e < 2000 and flag == 0:
                    for gtval in gtvalues:
                        x, y, w, h = result
                        iou = IoU(gtval, [x, x + w, y, y + h])

                        if counter < 30:
                            if iou > 0.70:
                                timage = imout[x : x + w, y : y + h]
                                resized = cv.resize(timage, (224, 224), interpolation = cv.INTER_AREA)
                                self.train_images.append(resized)
                                self.train_labels.append(1)
                                counter += 1
                        else :
                            fflag = 1
                        if falsecounter < 30:
                            if iou < 0.3:
                                timage = imout[x : x + w, y : y + h]
                                resized = cv.resize(timage, (224, 224), interpolation = cv.INTER_AREA)
                                self.train_images.append(resized)
                                self.train_labels.append(0)
                                falsecounter += 1
                        else :
                            bflag = 1
                    if fflag == 1 and bflag == 1:  
                        flag = 1

class SVMDataset(Dataset):
    def __init__(self, image_path, annot_path):
        self.image_path = image_path
        self.annot_path = annot_path
        self.ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.svm_images = []
        self.svm_labels = []
        self._load_data()

    def __getitem__(self, idx):
        image = self.svm_images[idx] / 255.0
        label = self.svm_labels[idx]
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image).float(), torch.tensor(label).float()

    def __len__(self):
        return len(self.svm_labels)

    def _load_data(self):
        for e, i in tqdm(enumerate(os.listdir(self.annot_path))):
            filename = i.split(".")[0] + ".jpg"
            image = cv.imread(os.path.join(self.image_path, filename))
            df = pd.read_csv(os.path.join(self.annot_path, i))
            gtvalues = []
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                gtvalues.append([x1, x2, y1, y2])
                timage = image[x1:x2,y1:y2]
                resized = cv.resize(timage, (224,224), interpolation = cv.INTER_AREA)
                self.svm_images.append(resized)
                self.svm_labels.append([0,1])

            self.ss.setBaseImage(image)
            self.ss.switchToSelectiveSearchFast()
            ssresults = self.ss.process()

            imout = image.copy()   
            counter = 0
            falsecounter = 0
            flag = 0

            for e, result in enumerate(ssresults):
                if e < 2000 and flag == 0:
                    for gtval in gtvalues:
                        x, y, w, h = result
                        iou = IoU(gtval, [x, x + w, y, y + h])

                        if falsecounter < 5:
                            if iou < 0.3:
                                timage = imout[x : x + w, y : y + h]
                                resized = cv.resize(timage, (224, 224), interpolation = cv.INTER_AREA)
                                self.svm_images.append(resized)
                                self.svm_labels.append([1, 0])
                                falsecounter += 1
                        else :
                            flag = 1
