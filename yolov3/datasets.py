
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils.iou import iou_width_height as iou

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir, self.label_dir, self.image_size = img_dir, label_dir, image_size
        self.transform, self.S, self.C = transform, S, C
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors_per_scale, self.num_anchors = self.anchors.shape[0] // 3, self.anchors.shape[0]
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path, img_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]), os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        bboxes, image = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist(), np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image, bboxes = augmentations["image"], augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                scale_idx, anchor_on_scale = anchor_idx // self.num_anchors_per_scale, anchor_idx % self.num_anchors_per_scale
                S, i, j = self.S[scale_idx], int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    x_cell, y_cell, width_cell, height_cell = S * x - j, S * y - i, width * S, height * S
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)
