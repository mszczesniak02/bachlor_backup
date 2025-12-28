# autopep8: off
import sys
import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

original_sys_path = sys.path.copy()
# moving to "segmentation/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))
# importing commons
from segmentation.common.dataloader import *
from segmentation.common.hparams import *
# importing utils
from utils.utils import *
# go back to the origin path
sys.path = original_sys_path
# normal imports

model_path_full_ds = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segmentation/yolo_big/runs/segment/yolov8m_crack_seg/weights/best.pt"
model_path_only_DeepCrack = r"/home/krzeslaav/Projects/bachlor/model_tests/ONLY_DEEPCRACK/yolo8/runs/segment/yolov8m_crack_seg/weights/best.pt"

def yolo_predict_single(model, dataset, index=0):
    """
    Runs inference on a single image using YOLOv8.
    """
    # 1. Get raw item from dataset
    # Dataset returns (tensor_img, tensor_mask)
    # Tensor img is normalized and CHW.
    t_img, t_msk = dataset[index]

    # 2. Convert to Numpy for YOLO input and Visualization
    # We MUST denormalize for YOLO because it expects standard image values (0-255 or 0-1) 
    # and "ten2np(..., denormalize=True)" gives us exactly that (RGB, 0-255).
    img_numpy = ten2np(t_img, denormalize=True)
    mask_gt = ten2np(t_msk) # Ground truth mask

    # 3. Run Inference
    # YOLO.predict can take numpy array
    results = model.predict(img_numpy, imgsz=512, verbose=False)

    # 4. Process Output
    # Create empty mask if no detection
    mask_pred = np.zeros((img_numpy.shape[0], img_numpy.shape[1]), dtype=np.float32)

    if results and results[0].masks is not None:
        # masks.data is [N, H, W] tensors
        data = results[0].masks.data

        # Resize if needed (YOLO output might be smaller or strictly 512)
        if data.shape[1:] != (img_numpy.shape[0], img_numpy.shape[1]):
            data = data.float()
            data = F.interpolate(data.unsqueeze(1), size=(img_numpy.shape[0], img_numpy.shape[1]),
                                 mode='bilinear', align_corners=False).squeeze(1)

        # Collapse all detected instances into one binary mask
        # data > 0.5 gives binary masks for instances.
        # Check any instance presence.
        mask_pred_tensor = torch.any(data > 0.5, dim=0).float()
        mask_pred = mask_pred_tensor.cpu().numpy()

    return img_numpy, mask_gt, mask_pred


def main():
    # Helper: Check paths
    if not os.path.exists(model_path_only_DeepCrack):
        print(f"Warning: Model not found at {model_path_only_DeepCrack}")

    model = YOLO(model_path_only_DeepCrack)

    dataset = dataset_get(img_path="../../../../datasets/dataset_segmentation/test_img/",
                          mask_path="../../../../datasets/dataset_segmentation/test_lab/", transform=val_transform)

    magic = 30

    # Custom predict loop for YOLO
    img, msk, out = yolo_predict_single(model, dataset, magic)

    # Use standard visualization
    visualize_model_output(img, msk, out, save_path=None)


if __name__ == "__main__":
    main()
