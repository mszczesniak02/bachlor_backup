#autopep8: off
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ultralytics import YOLO
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- CONFIGURATION ---
UNET_PATH = "/content/m_unet.pth"
SEGFORMER_PATH = "/content/m_segformer.pth"
YOLO8_PATH = "/content/m_yolo8.pth"

# Set up paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# To reach 'src' parent if needed, but 'segmentation' is in 'src'
src_dir = os.path.dirname(os.path.dirname(current_dir))
# Actually we are in src/segmentation, so we need to add 'src' to path to import segmentation.common
if os.path.abspath(os.path.join(current_dir, '../')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(current_dir, '../')))

from segmentation.common.hparams import *  # To get validation paths
from segmentation.common.dataloader import dataset_get, val_transform
from segmentation.common.model import model_load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512

def predict_torch(model, images):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        probs = torch.sigmoid(outputs)
    return probs.cpu().numpy()  # [B, 1, H, W]

def predict_yolo(model, images_tensor):
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=images_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=images_tensor.device).view(1, 3, 1, 1)
    imgs_denorm = torch.clamp(images_tensor * std + mean, 0, 1)

    # Predict
    results = model.predict(imgs_denorm, imgsz=IMAGE_SIZE, verbose=False)

    batch_masks = []
    h, w = images_tensor.shape[2], images_tensor.shape[3]

    for res in results:
        if res.masks is not None:
            data = res.masks.data
            if data.shape[1:] != (h, w):
                data = torch.nn.functional.interpolate(data.unsqueeze(1), size=(
                    h, w), mode='bilinear', align_corners=False).squeeze(1)

            mask_pred = torch.max(data, dim=0)[0]
        else:
            mask_pred = torch.zeros((h, w), device=images_tensor.device)
        batch_masks.append(mask_pred)

    if batch_masks:
        return torch.stack(batch_masks).unsqueeze(1).cpu().numpy()
    return np.zeros((images_tensor.shape[0], 1, h, w))

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculates metrics for a single image:
    Accuracy, Precision, Recall, F1 (Dice), IoU
    Note: Dice is equivalent to F1 score for binary segmentation.
    """
    pred_bin = (pred_mask > 0.5).astype(np.uint8).flatten()
    gt_bin = (gt_mask > 0.5).astype(np.uint8).flatten()

    # If both are empty, perfect match
    if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
        return {
            "mIoU": 1.0,
            "Dice": 1.0,
            "F1": 1.0,
            "Precision": 1.0,
            "Recall": 1.0,
            "Accuracy": 1.0
        }

    # If gt is empty but pred is not, IoU/Dice/F1/Precision = 0. Recall is undefined (handling as 1 or 0 -> usually 0 interest if no positive class).
    # We will use sklearn to handle edge cases properly where possible (zero_division=0/1)

    # Accuracy
    acc = accuracy_score(gt_bin, pred_bin)

    # Precision, Recall, F1
    precision = precision_score(gt_bin, pred_bin, zero_division=0)
    recall = recall_score(gt_bin, pred_bin, zero_division=0)
    f1 = f1_score(gt_bin, pred_bin, zero_division=0)

    # IoU
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    iou = intersection / union if union > 0 else 1.0 if np.sum(gt_bin) == 0 else 0.0

    return {
        "mIoU": iou,
        "Dice": f1, # Dice Coefficient is F1 Score for binary classification
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": acc
    }

def main():
    print(f"Running Metrics calculation on {DEVICE}")

    # 1. Load Models
    models = {}

    print("Loading U-Net...")
    try:
        models["U-Net"] = model_load("unet", filepath=UNET_PATH, device=DEVICE)
    except Exception as e:
        print(f"Failed to load U-Net: {e}")

    print("Loading SegFormer...")
    try:
        models["SegFormer"] = model_load(
            "segformer", filepath=SEGFORMER_PATH, device=DEVICE)
    except Exception as e:
        print(f"Failed to load SegFormer: {e}")

    print("Loading YOLOv8...")
    try:
        if os.path.exists(YOLO8_PATH):
            models["YOLOv8"] = YOLO(YOLO8_PATH)
        else:
            print(f"YOLOv8 weights not found at {YOLO8_PATH}")
    except Exception as e:
        print(f"Failed to load YOLOv8: {e}")

    if not models:
        print("No models loaded. Exiting.")
        return

    # 2. Load Validation Dataset
    print(f"Loading validation dataset metrics from:")
    print(f"Images: {IMG_TEST_PATH}")
    print(f"Masks: {MASK_TEST_PATH}")

    dataset = dataset_get(img_path=IMG_TEST_PATH,
                          mask_path=MASK_TEST_PATH, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=False, num_workers=4)

    print(f"Dataset size: {len(dataset)}")

    # 3. Evaluation Loop
    # Store lists of metrics dictionaries
    results = {name: [] for name in models.keys()}
    results["Ensemble"] = []

    for images, masks in tqdm(dataloader, desc="Evaluating"):
        images = images.to(DEVICE)
        masks = masks.numpy()  # [B, 1, H, W]

        batch_preds = {}

        # Get predictions for all models
        for name, model in models.items():
            if name == "YOLOv8":
                preds = predict_yolo(model, images) # [B, 1, H, W]
            else:
                preds = predict_torch(model, images) # [B, 1, H, W]
            batch_preds[name] = preds

        # Calculate Ensemble Prediction (Soft Voting / Average Probability)
        # Assuming equal weights for now as simpler start
        start_key = list(batch_preds.keys())[0]
        ensemble_pred = np.zeros_like(batch_preds[start_key])

        for name in batch_preds:
            ensemble_pred += batch_preds[name]
        ensemble_pred /= len(batch_preds)

        # Calculate Metrics for each item in batch
        for i in range(images.shape[0]):
            gt = masks[i, 0]

            # Individual Models
            for name in models.keys():
                metrics = calculate_metrics(batch_preds[name][i, 0], gt)
                results[name].append(metrics)

            # Ensemble
            ens_metrics = calculate_metrics(ensemble_pred[i, 0], gt)
            results["Ensemble"].append(ens_metrics)

    # 4. Aggregate Results
    final_metrics = {}
    metric_keys = ["mIoU", "Dice", "F1", "Precision", "Recall", "Accuracy"]

    for name, metrics_list in results.items():
        aggregated = {}
        for key in metric_keys:
            values = [m[key] for m in metrics_list]
            aggregated[key] = np.mean(values)
        final_metrics[name] = aggregated

    # 5. Print Table
    print("\n" + "="*95)
    print(f"{'Model':<15} | {'mIoU':<10} | {'Dice (F1)':<12} | {'Precision':<10} | {'Recall':<10} | {'Accuracy':<10}")
    print("-" * 95)

    for name, metrics in final_metrics.items():
        print(f"{name:<15} | {metrics['mIoU']:.4f}     | {metrics['Dice']:.4f}       | {metrics['Precision']:.4f}     | {metrics['Recall']:.4f}     | {metrics['Accuracy']:.4f}")
    print("="*95 + "\n")


if __name__ == "__main__":
    main()
