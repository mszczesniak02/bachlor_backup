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
import matplotlib.pyplot as plt
import concurrent.futures

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
UNET_PATH = "/content/model_unet.pth"
SEGFORMER_PATH = "/content/segformer_full.pth"
YOLO8_PATH = "/content/yolo8_deepcrack.pt"

# DATA PATHS
TEST_IMG = "/content/datasets/multi/test_img/"
TEST_LAB = "/content/datasets/multi/test_lab/"


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
        process = psutil.Process()
        ram_usage_mb = process.memory_info().rss / 1024**2
        gpu_usage_mb = 0
        if torch.cuda.is_available():
            gpu_usage_mb = torch.cuda.memory_allocated() / 1024**2
        return cpu_usage, ram_usage_mb, gpu_usage_mb

    def predict_torch(self, model, images, return_probs=False):
        """Standard PyTorch Inference"""
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.sigmoid(outputs)
        if return_probs:
            return probs.cpu().detach().numpy() # [B, 1, H, W]
        preds = (probs > 0.5).float()
        return preds.cpu().detach().numpy() # [B, 1, H, W]

    def predict_yolo(self, model, images_tensor, return_probs=False):
        """YOLO Inference"""
        preds_batch = []
        for i in range(images_tensor.shape[0]):
            img_tensor = images_tensor[i]
            img_numpy = ten2np(img_tensor, denormalize=True) 

            # Using 512 as requested
            results = model.predict(img_numpy, imgsz=512, verbose=False, conf=0.25)
            mask_pred = np.zeros((img_numpy.shape[0], img_numpy.shape[1]), dtype=np.float32)

            if results and results[0].masks is not None:
                data = results[0].masks.data
                if data.shape[1:] != (img_numpy.shape[0], img_numpy.shape[1]):
                    data = data.float()
                    data = F.interpolate(data.unsqueeze(1), size=(img_numpy.shape[0], img_numpy.shape[1]),
                                         mode='bilinear', align_corners=False).squeeze(1)

                if return_probs:
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

        ious, dices = [], []
        total_time, total_samples = 0, 0
        cpu_usages, ram_usages, gpu_usages = [], [], []

        np.random.seed(42)
        pbar = tqdm(dataloader, desc=f"Testing {model_name}")

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                start_time = time.time()

                if model_type == 'torch':
                    preds = self.predict_torch(model, images).squeeze(1) 
                elif model_type == 'yolo':
                    preds = self.predict_yolo(model, images) 

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

        result = {
            "Model": model_name,
            "IoU": np.mean(ious),
            "Dice": np.mean(dices),
            "Avg Inference Time (ms/img)": (total_time / total_samples) * 1000,
            "Total Time (s)": total_time,
            "Avg CPU (%)": np.mean(cpu_usages),
            "Max RAM (MB)": np.max(ram_usages),
            "Max GPU Mem (MB)": np.max(gpu_usages)
        }
        self.results.append(result)
        return result

    def worker_predict(self, model_info, images):
        name, m_type, model = model_info
        if m_type == 'torch':
            return self.predict_torch(model, images, return_probs=True).squeeze(1)
        elif m_type == 'yolo':
            return self.predict_yolo(model, images, return_probs=True)

    def evaluate_ensemble_parallel(self, model_specs, dataloader):
        """
        Parallel inference using ThreadPoolExecutor.
        model_specs: list of tuples (name, type, model)
        """
        name = "Ensemble Parallel"
        print(f"\n--- Benchmarking {name} ---")

        # Ensure all torch models are on device
        for n, t, m in model_specs:
            if t == 'torch':
                m.to(self.device).eval()

        ious, dices = [], []
        total_time, total_samples = 0, 0

        # We'll rely on global list for final stats, but can track locally too
        pbar = tqdm(dataloader, desc=f"Testing Ensemble Parallel")

        with torch.no_grad():
            # Use a ThreadPoolExecutor
            # Note: PyTorch CUDA operations release GIL, so this helps.
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_specs)) as executor:
                for batch_idx, (images, masks) in enumerate(pbar):
                    images = images.to(self.device)
                    start_time = time.time()

                    # Submit tasks
                    futures = [executor.submit(self.worker_predict, spec, images) for spec in model_specs]

                    # Collect results
                    batch_probs = []
                    for future in concurrent.futures.as_completed(futures):
                        batch_probs.append(future.result())

                    # Currently batch_probs order is random due to as_completed, 
                    # but for Averaging it DOES NOT WRONG.

                    avg_probs = np.mean(np.array(batch_probs), axis=0)
                    preds = (avg_probs > 0.5).astype(np.float32)

                    end_time = time.time()
                    total_time += (end_time - start_time)

                    masks_np = masks.cpu().numpy().squeeze(1)
                    for i in range(preds.shape[0]):
                        iou, dice = calculate_metrics(masks_np[i], preds[i])
                        ious.append(iou)
                        dices.append(dice)
                    total_samples += images.size(0)

        result = {
            "Model": name,
            "IoU": np.mean(ious),
            "Dice": np.mean(dices),
            "Avg Inference Time (ms/img)": (total_time / total_samples) * 1000,
            "Total Time (s)": total_time
        }
        self.results.append(result)
        return result

    def save_results(self, filename="segmentation_benchmark.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nBenchmark results saved to {filename}")
        print(df.to_string(index=False))

    def visualize_comparisons(self, model_specs, dataset, num_samples=5):
        print(f"\nGenerowanie {num_samples} obrazów porównawczych...")
        indices = np.random.choice(len(dataset), num_samples, replace=False)

        # Ensure models are ready
        for n, t, m in model_specs:
            if t == 'torch':
                m.to(self.device).eval()

        for idx in indices:
            img_tensor, mask_tensor = dataset[idx]
            # img_tensor: [3, H, W], mask_tensor: [1, H, W]

            # Prepare batch of 1
            images = img_tensor.unsqueeze(0).to(self.device)

            predictions = {}

            # Individual Predictions
            # Use same worker logic or sequential
            with torch.no_grad():
                for n, t, m in model_specs:
                    if t == 'torch':
                        pred = self.predict_torch(m, images, return_probs=False).squeeze()
                    elif t == 'yolo':
                        pred = self.predict_yolo(m, images, return_probs=False).squeeze()
                    predictions[n] = pred

                # Ensemble Prediction
                probs_list = []
                for n, t, m in model_specs:
                    if t == 'torch':
                        p = self.predict_torch(m, images, return_probs=True).squeeze()
                    elif t == 'yolo':
                        p = self.predict_yolo(m, images, return_probs=True).squeeze()
                    probs_list.append(p)

                ens_prob = np.mean(np.array(probs_list), axis=0)
                ens_pred = (ens_prob > 0.5).astype(np.float32)
                predictions["Ensemble"] = ens_pred

            # Plotting
            fig, ax = plt.subplots(1, 6, figsize=(24, 4))

            # Input
            ax[0].imshow(ten2np(img_tensor, denormalize=True))
            ax[0].set_title("Input")
            ax[0].axis('off')

            # GT
            ax[1].imshow(ten2np(mask_tensor, denormalize=False), cmap='gray')
            ax[1].set_title("Ground Truth")
            ax[1].axis('off')

            # Models
            names_map = ["U-Net", "SegFormer", "YOLOv8", "Ensemble"]
            # Map specs to simplistic names if needed, but here we can just iterate known order
            # model_specs order: UNet, SegFormer, YOLOv8

            # Mapping:
            # predictions keys: "U-Net", "SegFormer", "YOLOv8" (from passed names)

            for i, key in enumerate(names_map):
                ax_idx = i + 2
                if key in predictions:
                    ax[ax_idx].imshow(predictions[key], cmap='gray', vmin=0, vmax=1)
                    ax[ax_idx].set_title(key)
                else:
                    ax[ax_idx].text(0.5, 0.5, "Not Loaded", ha='center', va='center', fontsize=12)
                    ax[ax_idx].set_title(f"{key} (Missing)")
                ax[ax_idx].axis('off')

            plt.tight_layout()
            out_name = f"benchmark_sample_{idx}.png"
            plt.savefig(out_name)
            plt.close()
            print(f"Saved {out_name}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on: {device}")
    benchmark = Benchmark(device)

    # Init Dataset
    dataset = dataset_get(img_path=TEST_IMG, mask_path=TEST_LAB, transform=val_transform)
    dataloader = dataloader_get(dataset, is_training=False, bsize=16)

    # Load Models Globally
    models = []

    # U-Net
    try:
        if os.path.exists(UNET_PATH):
            m = model_load("unet", filepath=UNET_PATH, device=device)
            models.append(("U-Net", "torch", m))
            print("Loaded U-Net")
        else:
            print(f"FAILED TO LOAD U-Net: File not found at {UNET_PATH}")
            parent = os.path.dirname(UNET_PATH)
            if os.path.exists(parent):
                print(f"Contents of {parent}: {os.listdir(parent)}")
            else:
                print(f"Parent directory {parent} does not exist.")
    except Exception as e: print(f"Error loading U-Net: {e}")

    # SegFormer
    try:
        if os.path.exists(SEGFORMER_PATH):
            m = model_load("segformer", filepath=SEGFORMER_PATH, device=device)
            models.append(("SegFormer", "torch", m))
            print("Loaded SegFormer")
        else:
            print(f"FAILED TO LOAD SegFormer: File not found at {SEGFORMER_PATH}")
            parent = os.path.dirname(SEGFORMER_PATH)
            if os.path.exists(parent):
                print(f"Contents of {parent}: {os.listdir(parent)}")
            else:
                print(f"Parent directory {parent} does not exist.")
    except Exception as e: print(f"Error loading SegFormer: {e}")

    # YOLOv8
    try:
        if os.path.exists(YOLO8_PATH):
            m = YOLO(YOLO8_PATH)
            models.append(("YOLOv8", "yolo", m))
            print("Loaded YOLOv8")
        else:
            print(f"FAILED TO LOAD YOLOv8: File not found at {YOLO8_PATH}")
            parent = os.path.dirname(YOLO8_PATH)
            if os.path.exists(parent):
                print(f"Contents of {parent}: {os.listdir(parent)}")
            else:
                print(f"Parent directory {parent} does not exist.")
    except Exception as e: print(f"Error loading YOLOv8: {e}")

    if not models:
        print("No models loaded.")
        return

    # Benchmark Individual
    # Re-use loaded models
    for n, t, m in models:
        benchmark.evaluate_model(n, t, m, dataloader)

    # Benchmark Ensemble Parallel
    if len(models) >= 2:
        benchmark.evaluate_ensemble_parallel(models, dataloader)

    benchmark.save_results()

    # Visualize
    benchmark.visualize_comparisons(models, dataset, num_samples=5)

if __name__ == "__main__":
    main()
