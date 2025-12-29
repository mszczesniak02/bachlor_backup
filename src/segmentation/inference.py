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
YOLO8_PATH = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segmentation/yolo_big/runs/segment/yolov8m_crack_seg/weights/best.pt"
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

    # 5. Ensemble (Average of all valid predictions)
    ensemble_pred = None
    valid_preds = [p for name, p in predictions if p is not None]
    if valid_preds:
        # Stack and mean
        ensemble_pred = np.mean(np.array(valid_preds), axis=0)
        predictions.append(("Ensemble", ensemble_pred))
    else:
        predictions.append(("Ensemble", None))

    # VISUALIZATION
    # Rows: 4
    # Row 1: Image, Mask
    # Row 2: U-Net, SegFormer
    # Row 3: YOLOv8, YOLOv12
    # Row 4: Ensemble (Centered)

    rows = 4
    cols = 2
    fig = plt.figure(figsize=(10, 20))
    gs = fig.add_gridspec(rows, cols)

    # Row 1: Input and GT
    ax_img = fig.add_subplot(gs[0, 0])
    ax_mask = fig.add_subplot(gs[0, 1])

    # Display Image (denormalized)
    img_disp = ten2np(img_tensor, denormalize=True)
    ax_img.imshow(img_disp)
    ax_img.set_title("Input Image")
    ax_img.axis('off')

    # Display GT Mask
    mask_disp = ten2np(mask_tensor)
    ax_mask.imshow(mask_disp, cmap='gray')
    ax_mask.set_title("Ground Truth")
    ax_mask.axis('off')

    # Models Mapping for Rows 2 and 3
    # predictions list index:
    # 0: UNet, 1: SegFormer, 2: YOLO8, 3: YOLO12, 4: Ensemble

    # Row 2
    ax_u = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])

    model_axes = [ax_u, ax_s]

    # Row 3
    ax_y8 = fig.add_subplot(gs[2, 0])
    ax_y12 = fig.add_subplot(gs[2, 1])

    model_axes.extend([ax_y8, ax_y12])

    # Plot individual models
    for i in range(4):
        name, pred = predictions[i]
        ax = model_axes[i]

        if pred is not None:
            ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"{name}")
        else:
            ax.text(0.5, 0.5, "Not Found", ha='center')
            ax.set_title(f"{name}")
        ax.axis('off')

    # Row 4: Ensemble (Centered items)
    # Spanning both columns for key focus
    ax_ens = fig.add_subplot(gs[3, :]) # span all cols

    ens_name, ens_pred = predictions[4]
    if ens_pred is not None:
        ax_ens.imshow(ens_pred, cmap='gray', vmin=0, vmax=1)
        ax_ens.set_title(f"{ens_name} (Average)")
    else:
        ax_ens.text(0.5, 0.5, "No Ensemble", ha='center')
        ax_ens.set_title(f"{ens_name}")
    ax_ens.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
