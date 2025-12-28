
from sklearn.metrics import f1_score, jaccard_score
#autopep8:off
import prediction  # To reuse model loading functions
import sys
import os
import time
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import psutil
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
# Add src to path for imports
from hparams import *
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))
from utils import utils

from entry_processing.dataloader import EntryDataset
from classification.common.dataloader import CrackDataset as ClassDataset
from segmentation.common.dataloader import CrackDataset as SegDataset
from torch.utils.data import DataLoader

# Import Dataloaders - handle path differences

# Metrics


def measure_resources():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # MB


def benchmark_segmentation(model, model_name, dataloader, device):
    print(f"Benchmarking Segmentation: {model_name}")

    if model_name != "yolo":
        model.eval()

    ious = []
    times = []
    peak_mem = 0

    start_all = time.time()

    MAX_BATCHES = 50
    # YOLO requires specific handling, so we iterate differently or handle inside loop
    for i, (images, masks) in enumerate(tqdm(dataloader)):
        if i >= MAX_BATCHES:
            break
        # images: Tensor [B, 3, H, W]
        # masks: Tensor [B, 1, H, W]

        start_batch = time.time()

        if model_name == "yolo":
            # YOLO expects numpy images (H, W, 3) or file paths
            # We need to convert batch tensors back to list of numpy images
            # Check if dataloader returns normalized images. If A.Normalize was used, we must denormalize for YOLO?
            # Actually YOLO model call handles many formats. Let's convert to numpy [B, H, W, 3] uint8

            # Undo normalization (approximate if we don't have exact params easily accessible here,
            # but usually YOLO expects 0-255 RGB/BGR)
            # Assuming standard imagenet mean/std used in transform:
            # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

            imgs_np = images.permute(0, 2, 3, 1).cpu().numpy()

            # De-normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            imgs_np = std * imgs_np + mean
            imgs_np = np.clip(imgs_np * 255, 0, 255).astype(np.uint8)

            preds_batch = []
            # Predict batch
            # YOLO v8 predict supports list of images
            results = model.predict(
                [img for img in imgs_np],
                imgsz=512,
                verbose=False,
                device=0 if device == 'cuda' else 'cpu'
            )

            for res in results:
                if res.masks is not None:
                    # Get mask, resize if needed
                    # masks.data is [N, H, W] where N is number of objects
                    m = res.masks.data
                    if m.shape[0] > 0:
                        # Combine masks
                        combined = torch.any(
                            m > 0.5, dim=0).float().cpu().numpy()
                        # Resize to match ground truth 256x256 (val_transform size)
                        # Ground truth masks are 256x256 from dataloader
                        combined = cv2.resize(
                            combined, (256, 256), interpolation=cv2.INTER_NEAREST)
                    else:
                        combined = np.zeros((256, 256), dtype=np.float32)
                else:
                    combined = np.zeros((256, 256), dtype=np.float32)
                preds_batch.append(combined)

            preds = np.array(preds_batch)  # [B, 256, 256]

        else:
            images = images.to(device)
            with torch.no_grad():
                out = model(images)
                # Segformer outputs might be 1/4 size (64x64 if input 256)
                if out.shape[-1] != 256:
                    out = F.interpolate(out, size=(
                        256, 256), mode='bilinear', align_corners=False)

                preds = (torch.sigmoid(out) > 0.5).float().squeeze(
                    1).cpu().numpy()

        end_batch = time.time()
        times.append(end_batch - start_batch)

        # Calculate Metric
        masks_np = masks.squeeze(1).cpu().numpy()  # [B, 256, 256]

        for j in range(len(preds)):
            # Jaccard Score (IoU) for binary class
            iou = jaccard_score(masks_np[j].flatten(
            ) > 0.5, preds[j].flatten() > 0.5, zero_division=1)
            ious.append(iou)

        current_mem = measure_resources()
        if current_mem > peak_mem:
            peak_mem = current_mem

    total_time = time.time() - start_all
    avg_time = np.mean(times) if times else 0
    avg_iou = np.mean(ious) if ious else 0

    return {
        "model": model_name,
        "task": "segmentation",
        "metric_name": "IoU",
        "metric_value": avg_iou,
        "avg_inference_time_batch_s": avg_time,
        "total_time_s": total_time,
        "peak_ram_mb": peak_mem
    }


def benchmark_classification(model, model_name, dataloader, device):
    print(f"Benchmarking Classification: {model_name}")
    model.eval()
    all_preds = []
    all_labels = []
    times = []
    peak_mem = 0

    start_all = time.time()

    MAX_BATCHES = 50
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            if i >= MAX_BATCHES:
                break
            images = images.to(device)

            start_batch = time.time()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            end_batch = time.time()
            times.append(end_batch - start_batch)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            current_mem = measure_resources()
            if current_mem > peak_mem:
                peak_mem = current_mem

    total_time = time.time() - start_all
    avg_time = np.mean(times) if times else 0

    # Calculate F1
    # Check if we have labels. If dummy dataloader, labels might be 0
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        "model": model_name,
        "task": "classification",
        "metric_name": "F1",
        "metric_value": f1,
        "avg_inference_time_batch_s": avg_time,
        "total_time_s": total_time,
        "peak_ram_mb": peak_mem
    }


def benchmark_entry(model, model_name, dataloader, device):
    # Entry processing is binary classification (crack vs no_crack)
    return benchmark_classification(model, model_name, dataloader, device)


if __name__ == "__main__":
    # Settings
    BATCH_SIZE = 8  # Safe size
    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")

    print(f"Devices to benchmark: {devices_to_test}")

    all_results = []

    for device in devices_to_test:
        print(f"\n{'='*20}\nRunning benchmark on {device}\n{'='*20}")

        # --- 1. Load Models ---
        print("\n--- Loading Models ---")
        models_dict = {}

        # Segmentation
        try:
            models_dict['segformer'] = prediction.load_segformer(device=device)
        except Exception as e:
            print(f"Failed to load SegFormer: {e}")

        try:
            models_dict['unet'] = prediction.load_unet(device=device)
        except Exception as e:
            print(f"Failed to load U-Net: {e}")

        try:
            models_dict['yolo'] = prediction.load_yolo(device=device)
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            models_dict['yolo'] = None

        # Classification
        try:
            models_dict['efficientnet'] = prediction.load_efficientnet(device=device)
        except Exception as e:
            print(f"Failed to load EfficientNet: {e}")

        try:
            models_dict['convnext'] = prediction.load_convnext(device=device)
        except Exception as e:
            print(f"Failed to load ConvNeXt: {e}")

        # Entry
        try:
            models_dict['entry_model'] = prediction.load_domain_controller(
                device=device)
        except Exception as e:
            print(f"Failed to load Entry Model: {e}")
            models_dict['entry_model'] = None

        # --- 2. Datasets & Dataloaders ---
        # Datasets are loaded once, but we'll recreate dataloaders just in case or reuse
        # Better to reuse dataloader definitions but they are device agnostic usually
        # (Move to device happens inside loop)

        # ... (Reusing dataset loading code from global scope or function if needed, 
        # but simpler to just init here or before loop. 
        # Dataset init involves file I/O, better do once before loop if possible, 
        # but benchmark structure in original code had it inside main. 
        # I will keep the imports and path checks outside loop or suppressed for brevity in this replace)

        print("\n--- Preparing Datasets ---")
        from segmentation.common.hparams import IMG_TEST_PATH, MASK_TEST_PATH
        from classification.common.hparams import TEST_DIR as CLF_TEST_DIR
        # Import user-defined paths from local hparams
        from hparams import SEG_IMG_TEST_PATH, SEG_MASK_TEST_PATH, CLASS_IMG_TEST_PATH_ROOT
        from entry_processing.hparams import TEST_DIR as ENTRY_TEST_DIR

        # Override with manual paths if available in hparams (user provided)
        try:
            # These are imported from hparams at top of file, but let's use them locally
            seg_img_path = SEG_IMG_TEST_PATH
            seg_mask_path = SEG_MASK_TEST_PATH
            clf_test_path = CLASS_IMG_TEST_PATH_ROOT
            print("Using paths from hparams.py")
        except NameError:
            print("Manual paths not found in hparams, using defaults.")
            seg_img_path = IMG_TEST_PATH
            seg_mask_path = MASK_TEST_PATH
            clf_test_path = CLF_TEST_DIR

        # Check if paths exist
        print(f"Seg Test Path: {seg_img_path}")
        print(f"Clf Test Path: {clf_test_path}")
        print(f"Entry Test Path: {ENTRY_TEST_DIR}")
        # Segmentation Dataloader
        seg_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], is_check_shapes=False)

        if os.path.exists(seg_img_path) and os.path.exists(seg_mask_path):
            seg_ds = SegDataset(img_dir=seg_img_path,
                                mask_dir=seg_mask_path, transform=seg_transform)
            seg_loader = DataLoader(
                seg_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        else:
            print(f"Segmentation dataset not found at {seg_img_path}")
            seg_loader = None

        # Classification Dataloader
        clf_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        if os.path.exists(clf_test_path):
            class_ds = ClassDataset(root_dir=clf_test_path, transform=clf_transform)
            class_loader = DataLoader(
                class_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        else:
            print(f"Classification dataset not found at {clf_test_path}")
            class_loader = None

        # Entry Dataloader
        entry_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        if os.path.exists(ENTRY_TEST_DIR):
            entry_ds = EntryDataset(root_dir=ENTRY_TEST_DIR,
                                    transform=entry_transform)
            entry_loader = DataLoader(
                entry_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        else:
            print(f"Entry dataset not found at {ENTRY_TEST_DIR}")
            entry_loader = None

        # --- 3. Benchmarking ---
        print("\n--- Starting Benchmarks ---")

        # Segmentation
        if seg_loader:
            if models_dict.get('unet'):
                res = benchmark_segmentation(
                    models_dict['unet'], "U-Net", seg_loader, device)
                res['device'] = device
                all_results.append(res)

            if models_dict.get('segformer'):
                res = benchmark_segmentation(
                    models_dict['segformer'], "SegFormer", seg_loader, device)
                res['device'] = device
                all_results.append(res)

            if models_dict.get('yolo'):
                res = benchmark_segmentation(
                    models_dict['yolo'], "yolo", seg_loader, device)
                res['device'] = device
                all_results.append(res)

        # Classification
        if class_loader:
            if models_dict.get('efficientnet'):
                res = benchmark_classification(
                    models_dict['efficientnet'], "EfficientNet", class_loader, device)
                res['device'] = device
                all_results.append(res)

            if models_dict.get('convnext'):
                res = benchmark_classification(
                    models_dict['convnext'], "ConvNeXt", class_loader, device)
                res['device'] = device
                all_results.append(res)

        # Entry
        if entry_loader and models_dict.get('entry_model'):
            res = benchmark_entry(
                models_dict['entry_model'], "EntryController", entry_loader, device)
            res['device'] = device
            all_results.append(res)

    # --- 4. Save Results ---
    if all_results:
        df = pd.DataFrame(all_results)
        print("\nResults Summary:")
        print(df)
        df.to_csv("benchmark_results.csv", index=False)
        print("Benchmark results saved to benchmark_results.csv")
    else:
        print("No results generated.")
