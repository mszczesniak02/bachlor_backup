#autopep8: off
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from ultralytics import YOLO

original_sys_path = sys.path.copy()

# moving to "segmentation/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

# importing commons
from segmentation.common.dataloader import *
from segmentation.common.model import *
from segmentation.common.hparams import *

# importing utils
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path

# MODEL PATHS
UNET_PATH = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/model_unet_0.5960555357910763.pth"
SEGFORMER_PATH = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segformermodel_segformer_0.5864474233337809.pth"
YOLO8_PATH = "/home/krzeslaav/Desktop/uuwe/yolo8_max/runs/segment/yolov8m_crack_seg/weights/best.pt"
YOLO12_PATH = "/home/krzeslaav/Projects/bachlor/src/segmentation/yolo_12_seg/runs/segment/yolov12m_crack_seg/weights/best.pt"

# DATA PATHS
TEST_IMG = "../../../datasets/dataset_segmentation/test_img/"
TEST_LAB = "../../../datasets/dataset_segmentation/test_lab/"

def predict_torch(model, image_tensor):
    """
    Returns prediction [H, W] normalized 0-1 (sigmoid)
    """
    output = model(image_tensor.unsqueeze(0).to(DEVICE))
    if isinstance(output, tuple): 
        output = output[0]

    output = torch.sigmoid(output)
    output = output.squeeze().cpu().detach().numpy()
    return output

def predict_yolo(model, image_tensor):
    """
    Returns prediction [H, W] binary or float
    """
    # Denormalize for YOLO
    img_numpy = ten2np(image_tensor, denormalize=True) 

    results = model.predict(img_numpy, imgsz=256, verbose=False)

    mask_pred = np.zeros((img_numpy.shape[0], img_numpy.shape[1]), dtype=np.float32)

    if results and results[0].masks is not None:
        data = results[0].masks.data
        if data.shape[1:] != (img_numpy.shape[0], img_numpy.shape[1]):
            data = data.float()
            data = F.interpolate(data.unsqueeze(1), size=(img_numpy.shape[0], img_numpy.shape[1]),
                                 mode='bilinear', align_corners=False).squeeze(1)

        mask_pred_tensor = torch.any(data > 0.5, dim=0).float()
        mask_pred = mask_pred_tensor.cpu().numpy()

    return mask_pred

def main():
    magic = 4 # Sample index
    print(f"Visualizing sample index: {magic}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init Dataset
    dataset = dataset_get(img_path=TEST_IMG, mask_path=TEST_LAB, transform=val_transform)

    if magic >= len(dataset):
        print(f"Index {magic} out of range (len={len(dataset)}). Using 0.")
        magic = 0

    img_tensor, mask_tensor = dataset[magic]
    # mask_tensor is [1, H, W]

    # Store predictions
    predictions = []

    # 1. U-Net
    try:
        model_unet = model_load("unet", filepath=UNET_PATH, device=device)
        model_unet.eval()
        pred_unet = predict_torch(model_unet, img_tensor)
        predictions.append(("U-Net", pred_unet))
        del model_unet
    except Exception as e:
        print(f"Error U-Net: {e}")
        predictions.append(("U-Net", None))

    # 2. SegFormer
    try:
        model_seg = model_load("segformer", filepath=SEGFORMER_PATH, device=device)
        model_seg.eval()
        pred_seg = predict_torch(model_seg, img_tensor)
        predictions.append(("SegFormer", pred_seg))
        del model_seg
    except Exception as e:
        print(f"Error SegFormer: {e}")
        predictions.append(("SegFormer", None))

    # 3. YOLOv8
    try:
        if os.path.exists(YOLO8_PATH):
            model_yolo8 = YOLO(YOLO8_PATH)
            pred_y8 = predict_yolo(model_yolo8, img_tensor)
            predictions.append(("YOLOv8", pred_y8))
            del model_yolo8
        else:
            predictions.append(("YOLOv8", None))
    except Exception as e:
        print(f"Error YOLOv8: {e}")
        predictions.append(("YOLOv8", None))

    # 4. YOLOv12
    try:
        if os.path.exists(YOLO12_PATH):
            model_yolo12 = YOLO(YOLO12_PATH)
            pred_y12 = predict_yolo(model_yolo12, img_tensor)
            predictions.append(("YOLOv12", pred_y12))
            del model_yolo12
        else:
            predictions.append(("YOLOv12", None))
    except Exception as e:
        print(f"Error YOLOv12: {e}")
        predictions.append(("YOLOv12", None))

    # 5. Ensemble (Average of valid predictions: SegFormer, YOLOv8, U-Net)
    ensemble_pred = None
    target_models = {"U-Net", "SegFormer", "YOLOv8"}
    valid_preds = [p for name, p in predictions if p is not None and name in target_models]
    if valid_preds:
        # Stack and mean
        ensemble_pred = np.mean(np.array(valid_preds), axis=0)
        predictions.append(("Ensemble", ensemble_pred))
    else:
        predictions.append(("Ensemble", None))

    # VISUALIZATION
    # Row 1: Image, Mask, U-Net
    # Row 2: SegFormer, YOLOv8, YOLOv12

    # Prepare display images
    img_disp = ten2np(img_tensor, denormalize=True)
    mask_disp = ten2np(mask_tensor)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # --- Row 1 ---
    # 1. Input Image
    ax[0, 0].imshow(img_disp)
    ax[0, 0].set_title("Obraz wejściowy")
    ax[0, 0].axis('off')

    # 2. Ground Truth
    ax[0, 1].imshow(mask_disp, cmap='gray')
    ax[0, 1].set_title("Maska")
    ax[0, 1].axis('off')

    # 3. U-Net
    name_u, pred_u = predictions[0]
    if pred_u is not None:
        ax[0, 2].imshow(pred_u, cmap='gray', vmin=0, vmax=1)
        ax[0, 2].set_title(f"{name_u}")
    else:
        ax[0, 2].text(0.5, 0.5, "Not Found", ha='center')
        ax[0, 2].set_title(f"{name_u}")
    ax[0, 2].axis('off')

    # --- Row 2 ---
    # 4. SegFormer
    name_s, pred_s = predictions[1]
    if pred_s is not None:
        ax[1, 0].imshow(pred_s, cmap='gray', vmin=0, vmax=1)
        ax[1, 0].set_title(f"{name_s}")
    else:
        ax[1, 0].text(0.5, 0.5, "Not Found", ha='center')
        ax[1, 0].set_title(f"{name_s}")
    ax[1, 0].axis('off')

    # 5. YOLOv8
    name_y8, pred_y8 = predictions[2]
    if pred_y8 is not None:
        ax[1, 1].imshow(pred_y8, cmap='gray', vmin=0, vmax=1)
        ax[1, 1].set_title(f"{name_y8}")
    else:
        ax[1, 1].text(0.5, 0.5, "Not Found", ha='center')
        ax[1, 1].set_title(f"{name_y8}")
    ax[1, 1].axis('off')

    # 6. Ensemble (Replaces YOLOv12 in visualization)
    ens_name, ens_pred = predictions[4] # Ensemble is index 4
    if ens_pred is not None:
        ax[1, 2].imshow(ens_pred, cmap='gray', vmin=0, vmax=1)
        ax[1, 2].set_title(f"{ens_name} (Średnia)")
    else:
        ax[1, 2].text(0.5, 0.5, "Not Found", ha='center')
        ax[1, 2].set_title(f"{ens_name}")
    ax[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
