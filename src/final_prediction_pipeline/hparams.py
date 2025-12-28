from torch import cuda
import os
SEGFORMER_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segformermodel_segformer_0.5864474233337809.pth"
UNET_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/model_unet_0.5960555357910763.pth"
YOLO_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segmentation/yolo_big/runs/segment/yolov8m_crack_seg/weights/best.pt"
EFFICIENTNET_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/classification/efficientnet/model_f1_0.9171_epoch14.pth"
CONVNEXT_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/classification/convnext/model_f1_0.9258_epoch9.pth"
DOMAIN_CONTROLLER_PATH = r"/home/krzeslaav/Projects/bachlor/models/entry_classificator/best_model.pth"
# --- Config ---
OUTPUT_IMAGE_SIZE = 512
DEVICE = "cpu" if cuda.is_available() else "cpu"
NUM_CLASSES = 4
IMAGE_PATH_0 = r"/home/krzeslaav/Projects/bachlor/image_test_0.jpg"
IMAGE_PATH_1 = r"/home/krzeslaav/Projects/bachlor/image_test_1.jpg"

# --- Dataset ---
SEG_IMG_TEST_PATH = r"/home/krzeslaav/Projects/datasets/dataset_segmentation/test_img"
SEG_MASK_TEST_PATH = r"/home/krzeslaav/Projects/datasets/dataset_segmentation/test_lab"

CLASS_IMG_TEST_PATH_ROOT = r"/home/krzeslaav/Projects/datasets/classification_width/test_img"
