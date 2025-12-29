#autopep8: off
import sys
import os
import time
import psutil
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
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

# MODEL PATHS (Taken from inference scripts)
UNET_PATH = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/model_unet_0.5960555357910763.pth"
SEGFORMER_PATH = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segformermodel_segformer_0.5864474233337809.pth"
YOLO8_PATH = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segmentation/yolo_big/runs/segment/yolov8m_crack_seg/weights/best.pt"
# Resolved relative path for YOLO12 based on the file location
YOLO12_PATH = "/home/krzeslaav/Projects/bachlor/src/segmentation/yolo_12_seg/runs/segment/yolov12m_crack_seg/weights/best.pt"

# DATA PATHS
TEST_IMG = "../../../datasets/dataset_segmentation/test_img/"
TEST_LAB = "../../../datasets/dataset_segmentation/test_lab/"

def calculate_metrics(y_true, y_pred, smooth=1e-6):
    """
    Calculate IoU and Dice for numpy arrays (H, W) or (B, H, W)
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

    return iou, dice

class Benchmark:
    def __init__(self, device):
        self.device = device
        self.results = []

    def get_resource_usage(self):
        cpu_usage = psutil.cpu_percent()

        # Process specific RAM in MB
        process = psutil.Process()
        ram_usage_mb = process.memory_info().rss / 1024**2

        gpu_usage_mb = 0
        if torch.cuda.is_available():
            gpu_usage_mb = torch.cuda.memory_allocated() / 1024**2  # MB

        return cpu_usage, ram_usage_mb, gpu_usage_mb

    def predict_torch(self, model, images, return_probs=False):
        """
        Standard PyTorch Inference: Logits -> Sigmoid -> Threshold
        """
        outputs = model(images)
        # Check if output is tuple (some models return loss too)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.sigmoid(outputs)

        if return_probs:
            return probs.cpu().detach().numpy() # [B, 1, H, W]

        preds = (probs > 0.5).float()
        return preds.cpu().detach().numpy() # [B, 1, H, W]

    def predict_yolo(self, model, images_tensor, return_probs=False):
        """
        YOLO Inference. Requires denormalization.
        Iterates batch because YOLO.predict handles lists or single images best for our post-processing logic.
        """
        preds_batch = []

        # Iterate over batch
        for i in range(images_tensor.shape[0]):
            img_tensor = images_tensor[i] # [3, H, W]

            # Denormalize to get [H, W, 3] uint8/float numpy for YOLO
            # ten2py usually returns RGB HWC in 0-255 range if denormalize=True
            img_numpy = ten2np(img_tensor, denormalize=True) 

            results = model.predict(img_numpy, imgsz=256, verbose=False) # imgsz should match input size

            # Default empty
            mask_pred = np.zeros((img_numpy.shape[0], img_numpy.shape[1]), dtype=np.float32)

            if results and results[0].masks is not None:
                data = results[0].masks.data
                # Resize if needed
                if data.shape[1:] != (img_numpy.shape[0], img_numpy.shape[1]):
                    data = data.float()
                    data = F.interpolate(data.unsqueeze(1), size=(img_numpy.shape[0], img_numpy.shape[1]),
                                         mode='bilinear', align_corners=False).squeeze(1)

                # Combine instances
                if return_probs:
                    # Max probability across instances (soft merge)
                    mask_pred_tensor = torch.max(data, dim=0)[0].float()
                else:
                    mask_pred_tensor = torch.any(data > 0.5, dim=0).float()

                mask_pred = mask_pred_tensor.cpu().numpy()

            preds_batch.append(mask_pred)

        return np.array(preds_batch) # [B, H, W]

    def evaluate_model(self, model_name, model_type, model, dataloader):
        print(f"\n--- Benchmarking {model_name} ---")

        if model_type == 'torch':
            model.to(self.device).eval()

        ious = []
        dices = []

        total_time = 0
        cpu_usages = []
        ram_usages = []
        gpu_usages = []

        total_samples = 0

        # Pre-warm GPU if torch
        if model_type == 'torch' and torch.cuda.is_available():
            dummy = torch.randn(1, 3, 256, 256).to(self.device)
            with torch.no_grad():
                model(dummy)

        np.random.seed(42)

        pbar = tqdm(dataloader, desc=f"Testing {model_name}")

        # Disable gradients
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                # masks are [B, 1, H, W] float

                start_time = time.time()

                preds = None
                if model_type == 'torch':
                    preds = self.predict_torch(model, images) 
                    # preds is [B, 1, H, W]
                    preds = preds.squeeze(1) # [B, H, W]

                elif model_type == 'yolo':
                    # YOLO takes CPU images usually, logic handles migration
                    preds = self.predict_yolo(model, images) 
                    # preds is [B, H, W]

                end_time = time.time()
                total_time += (end_time - start_time)

                # Metrics
                # Masks is [B, 1, H, W], squeeze to [B, H, W]
                masks_np = masks.cpu().numpy().squeeze(1)

                # Calculate per item in batch
                for i in range(preds.shape[0]):
                    iou, dice = calculate_metrics(masks_np[i], preds[i])
                    ious.append(iou)
                    dices.append(dice)

                total_samples += images.size(0)

                c, r, g = self.get_resource_usage()
                cpu_usages.append(c)
                ram_usages.append(r)
                gpu_usages.append(g)

        avg_iou = np.mean(ious)
        avg_dice = np.mean(dices)
        avg_inference_time = (total_time / total_samples) * 1000

        result = {
            "Model": model_name,
            "IoU": avg_iou,
            "Dice": avg_dice,
            "Avg Inference Time (ms/img)": avg_inference_time,
            "Total Time (s)": total_time,
            "Avg CPU (%)": np.mean(cpu_usages),
            "Max RAM (MB)": np.max(ram_usages),
            "Max GPU Mem (MB)": np.max(gpu_usages)
        }

        self.results.append(result)

        print(f"\nResults for {model_name}:")
        print(f"IoU: {avg_iou:.4f}")
        print(f"Dice: {avg_dice:.4f}")
        print(f"Time: {avg_inference_time:.2f} ms/img")

        return result

    def evaluate_ensemble(self, model_specs, dataloader):
        """
        model_specs: list of tuples (name, type, model)
        """
        name = "Ensemble (" + "+".join([n for n, t, m in model_specs]) + ")"
        print(f"\n--- Benchmarking {name} ---")

        for n, t, m in model_specs:
            if t == 'torch':
                m.to(self.device).eval()

        ious = []
        dices = []

        total_time = 0
        cpu_usages = []
        ram_usages = []
        gpu_usages = []

        total_samples = 0

        np.random.seed(42)
        pbar = tqdm(dataloader, desc=f"Testing Ensemble")

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                start_time = time.time()

                batch_probs = []

                for n, t, m in model_specs:
                    if t == 'torch':
                        # returns [B, 1, H, W]
                        p = self.predict_torch(m, images, return_probs=True).squeeze(1) # -> [B, H, W]
                    elif t == 'yolo':
                        # returns [B, H, W]
                        p = self.predict_yolo(m, images, return_probs=True)
                    batch_probs.append(p)

                # Stack: [Models, B, H, W] -> Mean -> [B, H, W]
                avg_probs = np.mean(np.array(batch_probs), axis=0)

                # Threshold
                preds = (avg_probs > 0.5).astype(np.float32)

                end_time = time.time()
                total_time += (end_time - start_time)

                masks_np = masks.cpu().numpy().squeeze(1)
                for i in range(preds.shape[0]):
                    iou, dice = calculate_metrics(masks_np[i], preds[i])
                    ious.append(iou)
                    dices.append(dice)

                total_samples += images.size(0)

                c, r, g = self.get_resource_usage()
                cpu_usages.append(c)
                ram_usages.append(r)
                gpu_usages.append(g)

        avg_iou = np.mean(ious)
        avg_dice = np.mean(dices)
        avg_inference_time = (total_time / total_samples) * 1000

        result = {
            "Model": "Ensemble (UNet+Seg+YOLO8)",
            "IoU": avg_iou,
            "Dice": avg_dice,
            "Avg Inference Time (ms/img)": avg_inference_time,
            "Total Time (s)": total_time,
            "Avg CPU (%)": np.mean(cpu_usages),
            "Max RAM (MB)": np.max(ram_usages),
            "Max GPU Mem (MB)": np.max(gpu_usages)
        }

        self.results.append(result)
        print(f"\nResults for Ensemble:")
        print(f"IoU: {avg_iou:.4f}")
        print(f"Dice: {avg_dice:.4f}")
        print(f"Time: {avg_inference_time:.2f} ms/img")
        return result

    def save_results(self, filename="segmentation_benchmark.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nBenchmark results saved to {filename}")
        print("\nSummary Table:")
        print(df[["Model", "IoU", "Dice", "Avg Inference Time (ms/img)"]].to_string(index=False))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on: {device}")

    benchmark = Benchmark(device)

    # Init Dataset
    # We use batch_size=16 for speed, modify if OOM
    # Transform is val_transform handled by dataloader_get's internal usage of dataset_get
    dataset = dataset_get(img_path=TEST_IMG, mask_path=TEST_LAB, transform=val_transform)
    dataloader = dataloader_get(dataset, is_training=False, bsize=16) 

    # 1. Benchmark U-Net
    try:
        if os.path.exists(UNET_PATH):
            model_unet = model_load("unet", filepath=UNET_PATH, device=device)
            benchmark.evaluate_model("U-Net", "torch", model_unet, dataloader)
            del model_unet # Free memory
            torch.cuda.empty_cache()
        else:
            print(f"Skipping U-Net: File not found {UNET_PATH}")
    except Exception as e:
        print(f"Error U-Net: {e}")

    # 2. Benchmark SegFormer
    try:
        if os.path.exists(SEGFORMER_PATH):
            model_seg = model_load("segformer", filepath=SEGFORMER_PATH, device=device)
            benchmark.evaluate_model("SegFormer", "torch", model_seg, dataloader)
            del model_seg
            torch.cuda.empty_cache()
        else:
            print(f"Skipping SegFormer: File not found {SEGFORMER_PATH}")
    except Exception as e:
        print(f"Error SegFormer: {e}")

    # 3. Benchmark YOLOv8
    try:
        if os.path.exists(YOLO8_PATH):
            model_yolo8 = YOLO(YOLO8_PATH)
            benchmark.evaluate_model("YOLOv8", "yolo", model_yolo8, dataloader)
            del model_yolo8
        else:
            print(f"Skipping YOLOv8: File not found {YOLO8_PATH}")
    except Exception as e:
        print(f"Error YOLOv8: {e}")

    # 4. Benchmark YOLOv12
    try:
        # Check explicit path or fallback
        yolo12_p = YOLO12_PATH
        if not os.path.exists(yolo12_p):
            # Try fallback to just filename if user has it locally different
            yolo12_p = "yolov12m_crack_seg.pt" 

        if os.path.exists(yolo12_p) or True: # Try loading anyway if 'best.pt' is generic
            # If path assumes existing but file check failed, might fail. 
            # But let's trust the path first.
            if os.path.exists(YOLO12_PATH):
                model_yolo12 = YOLO(YOLO12_PATH)
                benchmark.evaluate_model("YOLOv12", "yolo", model_yolo12, dataloader)
                del model_yolo12
            else:
                print(f"Skipping YOLOv12: File not found {YOLO12_PATH}")
    except Exception as e:
        print(f"Error YOLOv12: {e}")

    # 5. Benchmark Ensemble (UNet + SegFormer + YOLOv8)
    print("\nPreparing Ensemble (UNet + SegFormer + YOLOv8)...")
    ensemble_models = []

    # Reload U-Net
    try:
        if os.path.exists(UNET_PATH):
            m_unet = model_load("unet", filepath=UNET_PATH, device=device)
            ensemble_models.append(("ExampleUNet", "torch", m_unet))
    except: pass

    # Reload SegFormer
    try:
        if os.path.exists(SEGFORMER_PATH):
            m_seg = model_load("segformer", filepath=SEGFORMER_PATH, device=device)
            ensemble_models.append(("ExampleSeg", "torch", m_seg))
    except: pass

    # Reload YOLOv8
    try:
        if os.path.exists(YOLO8_PATH):
            m_y8 = YOLO(YOLO8_PATH)
            ensemble_models.append(("ExampleY8", "yolo", m_y8))
    except: pass

    if len(ensemble_models) == 3:
        benchmark.evaluate_ensemble(ensemble_models, dataloader)
        # Cleanup
        for n, t, m in ensemble_models:
            del m
        torch.cuda.empty_cache()
    else:
        print(f"Skipping Ensemble: Not all models loaded (Got {len(ensemble_models)}/3)")

    benchmark.save_results()

if __name__ == "__main__":
    main()
