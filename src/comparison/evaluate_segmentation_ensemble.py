#autopep8: off
import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
PATH_UNET = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/model_unet_0.5960555357910763.pth"
PATH_SEGFORMER = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segformermodel_segformer_0.5864474233337809.pth"
PATH_YOLO = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segmentation/yolo_big/runs/segment/yolov8m_crack_seg/weights/best.pt"

TEST_IMG_PATH = "/home/krzeslaav/Projects/datasets/dataset_segmentation/test_img"
TEST_MASK_PATH = "/home/krzeslaav/Projects/datasets/dataset_segmentation/test_lab"
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from segmentation.common.model import model_load
from segmentation.common.dataloader import dataset_get, val_transform

def predict_torch(model, images):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        # Assuming output is logits, apply sigmoid
        probs = torch.sigmoid(outputs)
    return probs.cpu().numpy() # [B, 1, H, W]

def predict_yolo(model, images_tensor):
    # YOLO expects images in [0, 1] range? Or [0, 255]? 
    # Ultralytics handles normalization internally usually, but input to predict needs to be checked.
    # visualize.py denormalized before passing to YOLO. Let's do the same to be safe.

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=images_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images_tensor.device).view(1, 3, 1, 1)
    imgs_denorm = torch.clamp(images_tensor * std + mean, 0, 1)

    # YOLO predict logic
    results = model.predict(imgs_denorm, imgsz=IMAGE_SIZE, verbose=False, conf=0.25)

    batch_masks = []
    h, w = images_tensor.shape[2], images_tensor.shape[3]

    for res in results:
        if res.masks is not None:
            data = res.masks.data # [N, H, W]
            # Resize if needed (YOLO might return lower res masks)
            if data.shape[1:] != (h, w):
                data = torch.nn.functional.interpolate(data.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

            # Combine masks (max confidence/prob)
            mask_pred = torch.max(data, dim=0)[0] # [H, W]
        else:
            mask_pred = torch.zeros((h, w), device=images_tensor.device)
        batch_masks.append(mask_pred)

    if batch_masks:
        return torch.stack(batch_masks).unsqueeze(1).cpu().numpy() # [B, 1, H, W]

    return np.zeros((images_tensor.shape[0], 1, h, w))


def calculate_metrics(pred_mask, gt_mask):
    # Thresholding
    pred_bin = (pred_mask > 0.5).astype(np.uint8)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)

    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin) - intersection

    tp = intersection
    fp = np.sum(pred_bin) - tp
    fn = np.sum(gt_bin) - tp

    iou = intersection / (union + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return iou, precision, recall

def main():
    print(f"Running evaluation on {DEVICE}")

    # 1. Load Models
    print("Loading models...")
    try:
        model_unet = model_load("unet", filepath=PATH_UNET, device=DEVICE)
        model_segformer = model_load("segformer", filepath=PATH_SEGFORMER, device=DEVICE)
        model_yolo = YOLO(PATH_YOLO)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Load Dataset
    print("Loading dataset...")
    # Note: dataset_get returns a Dataset object, we need a DataLoader
    dataset = dataset_get(img_path=TEST_IMG_PATH, mask_path=TEST_MASK_PATH, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    print(f"Dataset size: {len(dataset)}")

    # 3. Evaluation Loop
    total_iou = 0
    total_precision = 0
    total_recall = 0
    num_batches = 0

    # Store per-model metrics if needed, but request was for Ensemble

    pbar = tqdm(dataloader, desc="Evaluating Ensemble")
    for images, masks in pbar:
        images = images.to(DEVICE)
        masks = masks.numpy() # [B, 1, H, W]

        # Inference
        prob_unet = predict_torch(model_unet, images)     # [B, 1, H, W]
        prob_seg  = predict_torch(model_segformer, images) # [B, 1, H, W]
        prob_yolo = predict_yolo(model_yolo, images)       # [B, 1, H, W]

        # Ensemble (Soft Voting)
        prob_ensemble = (prob_unet + prob_seg + prob_yolo) / 3.0

        # Calculate Metrics for batch
        # We can calculate for the whole batch at once using numpy operations or iterate
        # Let's iterate to be safe with metric definitions

        batch_iou = 0
        batch_prec = 0
        batch_rec = 0

        for i in range(images.shape[0]):
            iou, prec, rec = calculate_metrics(prob_ensemble[i, 0], masks[i, 0])
            batch_iou += iou
            batch_prec += prec
            batch_rec += rec

        total_iou += batch_iou
        total_precision += batch_prec
        total_recall += batch_rec
        num_batches += images.shape[0]

        # Update progress bar
        current_iou = total_iou / num_batches
        pbar.set_postfix({"Avg IoU": f"{current_iou:.4f}"})

    # 4. Final Results
    avg_iou = total_iou / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches

    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)

    print("\n" + "="*40)
    print("ENSEMBLE SEGMENTATION RESULTS (Validation Set)")
    print("="*40)
    print(f"Models: UNet + SegFormer + YOLOv8")
    print("-" * 40)
    print(f"IoU (Intersection over Union) : {avg_iou:.4f}")
    print(f"Precision                     : {avg_precision:.4f}")
    print(f"Recall                        : {avg_recall:.4f}")
    print(f"F1-Score (Dice)               : {f1_score:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
