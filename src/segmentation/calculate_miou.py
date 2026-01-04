#autopep8: off
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ultralytics import YOLO

# --- CONFIGURATION ---
# Using the same paths as in evaluate_segmentation_ensemble.py or hparams if possible.
# But hardcoding known good paths from previous tools for reliability as requested.

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


def calculate_miou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0.5).astype(np.uint8)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)

    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)

    if union == 0:
        return 1.0  # If both empty, it's a perfect match

    return intersection / union


def main():
    print(f"Running mIoU calculation on {DEVICE}")

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
    # Using IMG_TEST_PATH and MASK_TEST_PATH from hparams which corresponds to 'test' dataset,
    # but based on hparams naming it seems to be used for validation/testing.
    # The user request asks for "validacyjny zestaw danych", usually that's the one we test on.
    print(f"Loading validation dataset from:")
    print(f"Images: {IMG_TEST_PATH}")
    print(f"Masks: {MASK_TEST_PATH}")

    dataset = dataset_get(img_path=IMG_TEST_PATH,
                          mask_path=MASK_TEST_PATH, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=False, num_workers=4)

    print(f"Dataset size: {len(dataset)}")

    # 3. Evaluation Loop
    ious = {name: [] for name in models.keys()}

    for images, masks in tqdm(dataloader, desc="Evaluating"):
        images = images.to(DEVICE)
        masks = masks.numpy()  # [B, 1, H, W]

        for name, model in models.items():
            if name == "YOLOv8":
                preds = predict_yolo(model, images)
            else:
                preds = predict_torch(model, images)

            # Calculate IoU for each image in batch
            for i in range(images.shape[0]):
                iou = calculate_miou(preds[i, 0], masks[i, 0])
                ious[name].append(iou)

    # 4. Results
    print("\n" + "="*30)
    print("mIoU Results (Validation Set)")
    print("="*30)
    for name, iou_list in ious.items():
        miou = np.mean(iou_list)
        print(f"{name:10s}: {miou:.4f}")
    print("="*30)


if __name__ == "__main__":
    main()
