import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
# from utils import seed_everything

DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = f"{DATASET}/images/"
LABEL_DIR = f"{DATASET}/labels/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Rescaled to [0,1]

# Rescale augmentation parameters
scale = 1.1
resize = A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale))
padding = A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        )
crop = A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE)
color = A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4)
affine = A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Affine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        )
flip = A.HorizontalFlip(p=0.5)
blur = A.Blur(p=0.1)
clahe = A.CLAHE(p=0.1)
posterize = A.Posterize(p=0.1)
to_gray = A.ToGray(p=0.1)
channel_shuffle = A.ChannelShuffle(p=0.05)

# Augmentation pipeline
train_transforms = A.Compose(
    [
        resize,
        padding,
        crop,
        color,
        affine,
        flip,
        blur,
        clahe,
        posterize,
        to_gray,
        channel_shuffle,
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

# Test transformation pipeline
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
        min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
