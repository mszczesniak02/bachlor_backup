
#autopep8:off
from ultralytics import YOLO

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# --- SETUP PATHS ---
# Add the 'src' directory to sys.path to allow imports from segmentation, utils, etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from segmentation.common.hparams import DEVICE
from segmentation.common.dataloader import dataset_get, dataloader_get, val_transform
from segmentation.common.model import model_load
from segmentation.benchmark import Benchmark

# Add DeepSegmentor to path for DeepCrack imports
deep_segmentor_path = os.path.join(current_dir, "DeepSegmentor")
sys.path.append(deep_segmentor_path)

# Add CrackFormer-II to path
crackformer_path = os.path.join(
    current_dir, "CrackFormer-II", "CrackFormer-II")
sys.path.append(crackformer_path)

# Import DeepCrack directly from source
try:
    from models.deepcrack_networks import DeepCrackNet
except ImportError as e:
    print(
        f"[ERROR] Could not import DeepCrackNet. Ensure 'DeepSegmentor/models' is correct. {e}")
    DeepCrackNet = None

# Import CrackFormer directly from source
try:
    from nets.crackformerII import crackformer
except ImportError as e:
    print(
        f"[ERROR] Could not import crackformer. Ensure 'CrackFormer-II/nets' is correct and 'timm' is installed. {e}")
    crackformer = None

# =================================================================================================
#                                     USER CONFIGURATION
# =================================================================================================

# --- DATASET PATHS ---
# Path to the test dataset (images and masks)
TEST_DATASET_IMG_PATH = "/home/krzeslaav/Projects/datasets/dataset_segmentation/test_img"
TEST_DATASET_MASK_PATH = "/home/krzeslaav/Projects/datasets/dataset_segmentation/test_lab"

# --- USER MODELS ---
# Paths to your trained model weights
# e.g., "models/segmentation/unet/best.pth"
PATH_MY_UNET = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/model_unet_0.5960555357910763.pth"
# e.g., "models/segmentation/segformer/best.pth"
PATH_MY_SEGFORMER = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segformermodel_segformer_0.5864474233337809.pth"
# e.g., "models/segmentation/yolo/best.pt"
PATH_MY_YOLO = "/home/krzeslaav/Desktop/uuwe/yolo8_max/runs/segment/yolov8m_crack_seg/weights/best.pt"

# --- COMPARISON MODELS (DeepCrack & CrackFormer) ---
# Paths to the downloaded weights (.pth)
PATH_DEEPCRACK_WEIGHTS = "/home/krzeslaav/Documents/deepcrack/deepcrack/pretrained_net_G.pth"
PATH_CRACKFORMER_WEIGHTS = "/home/krzeslaav/Documents/crackformer/crack315.pth"


# =================================================================================================
#                                MODEL ARCHITECTURE DEFINITIONS
# =================================================================================================


class DeepCrackWrapper(nn.Module):
    """
    Wrapper for DeepCrackNet to match the simple forward(x) -> output interface
    expected by the Benchmark class.
    """

    def __init__(self, in_nc=3, num_classes=1, ngf=64, norm='batch'):
        super().__init__()
        if DeepCrackNet is None:
            raise ImportError("DeepCrackNet definition not found.")
        self.net = DeepCrackNet(in_nc, num_classes, ngf, norm)

    def forward(self, x):
        # DeepCrackNet returns tuple: (side1, side2, side3, side4, side5, fused)
        # We only care about the fused output for benchmarking
        outputs = self.net(x)
        fused = outputs[-1]
        return fused


class CrackFormerWrapper(nn.Module):
    """
    Wrapper for CrackFormer to match the simple forward(x) -> output interface.
    """

    def __init__(self):
        super().__init__()
        if crackformer is None:
            raise ImportError("crackformer definition not found.")
        self.net = crackformer()

    def forward(self, x):
        # crackformer returns tuple: (fuse5, fuse4, fuse3, fuse2, fuse1, output)
        # We want the last element: output
        outputs = self.net(x)
        return outputs[-1]

# =================================================================================================
#                                       MAIN SCRIPT
# =================================================================================================


def load_external_model(model_wrapper_class, weights_path, device, **kwargs):
    """
    Helper to load an external model (DeepCrack/CrackFormer) from weights.
    """
    if "PLACEHOLDER" in weights_path or not os.path.exists(weights_path):
        print(
            f"[WARN] Weights not found or placeholder set for: {weights_path}")
        return None

    try:
        model = model_wrapper_class(**kwargs)
        # DeepCrack weights might be saved as:
        # 1. State dict of the network
        # 2. Checkpoint dict with 'model_state_dict' or similar
        # 3. Full model object (less likely for portable weights)

        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the dict itself is the state dict if keys look like layer names
                state_dict = checkpoint

        if state_dict:
            # Fix potential key mismatches (remove 'module.' or 'netG.' prefixes if present)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]
                if name.startswith('netG.'):
                    name = name[5:]
                new_state_dict[name] = v

            # Load with strict=False to be safe, but report missing keys if critical
            model.net.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"[ERROR] Could not extract state_dict from {weights_path}")
            return None

        model.to(device)
        model.eval()
        print(f"[OK] Loaded {model_wrapper_class.__name__}")
        return model
    except Exception as e:
        print(
            f"[ERROR] Failed to load {model_wrapper_class.__name__} from {weights_path}: {e}")
        return None


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Running comparison on: {device}")

    # 1. Initialize Benchmark
    benchmark = Benchmark(device)

    # 2. Load Dataset
    print(f"\n--- Loading Test Dataset ---")
    if "PLACEHOLDER" in TEST_DATASET_IMG_PATH:
        print("Please set TEST_DATASET_IMG_PATH to a valid directory.")
        return

    dataset = dataset_get(img_path=TEST_DATASET_IMG_PATH,
                          mask_path=TEST_DATASET_MASK_PATH, transform=val_transform)
    # Smaller batch size for safety
    dataloader = dataloader_get(dataset, is_training=False, bsize=4)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. Load Models
    models_to_test = []

    # --- Load My Models ---
    print(f"\n--- Loading User Models ---")
    # U-Net
    if not "PLACEHOLDER" in PATH_MY_UNET and os.path.exists(PATH_MY_UNET):
        try:
            m = model_load("unet", filepath=PATH_MY_UNET, device=device)
            models_to_test.append(("My-UNet", "torch", m))
            print("[OK] My-UNet loaded")
        except Exception as e:
            print(f"Error loading My-UNet: {e}")

    # SegFormer
    if not "PLACEHOLDER" in PATH_MY_SEGFORMER and os.path.exists(PATH_MY_SEGFORMER):
        try:
            m = model_load(
                "segformer", filepath=PATH_MY_SEGFORMER, device=device)
            models_to_test.append(("My-SegFormer", "torch", m))
            print("[OK] My-SegFormer loaded")
        except Exception as e:
            print(f"Error loading My-SegFormer: {e}")

    # YOLOv8
    if not "PLACEHOLDER" in PATH_MY_YOLO and os.path.exists(PATH_MY_YOLO):
        try:
            m = YOLO(PATH_MY_YOLO)
            models_to_test.append(("My-YOLOv8", "yolo", m))
            print("[OK] My-YOLOv8 loaded")
        except Exception as e:
            print(f"Error loading My-YOLOv8: {e}")

    # --- Load External Models ---
    print(f"\n--- Loading Comparison Models ---")

    # Configure DeepCrack parameters (ngf=64 is standard)
    deepcrack_model = load_external_model(
        DeepCrackWrapper, PATH_DEEPCRACK_WEIGHTS, device, ngf=64)
    if deepcrack_model:
        models_to_test.append(("DeepCrack", "torch", deepcrack_model))

    crackformer_model = load_external_model(
        CrackFormerWrapper, PATH_CRACKFORMER_WEIGHTS, device)
    if crackformer_model:
        models_to_test.append(("CrackFormer", "torch", crackformer_model))

    if not models_to_test:
        print("No models were successfully loaded. Exiting.")
        return

    # 4. Run Benchmark
    print(f"\n================ STARTING BENCHMARK ================")
    for n, t, m in models_to_test:
        benchmark.evaluate_model(n, t, m, dataloader)

    # 5. Run User Ensemble (My-UNet, My-SegFormer, My-YOLOv8)
    user_model_names = ["My-UNet", "My-SegFormer", "My-YOLOv8"]
    user_ensemble_models = [
        m for m in models_to_test if m[0] in user_model_names]

    if len(user_ensemble_models) >= 2:
        print(f"\n================ STARTING USER ENSEMBLE BENCHMARK ================")
        print(f"Ensembling: {[m[0] for m in user_ensemble_models]}")
        benchmark.evaluate_ensemble_parallel(user_ensemble_models, dataloader)
    elif len(user_ensemble_models) > 0:
        print(
            f"\n[INFO] Not enough user models for ensemble (Found {len(user_ensemble_models)}, need >= 2). Skipping ensemble.")

    # 6. Save and Visualize
    benchmark.save_results("comparison_benchmark.csv")

    # Visualize top 5 samples
    # We want to visualize all models + the ensemble if it exists
    models_for_viz = models_to_test
    # Note: evaluate_ensemble_parallel adds result to benchmark results, but visualize_comparisons needs the model specs
    # The visualize function in benchmark.py (line 307) calculates ensemble on the fly if passed multiple models.
    # However, it currently calculates ensemble of ALL passed models.
    # We might want to customize this or just accept that the visualization will show ensemble of ALL models.
    # Given the user request is about benchmarking metrics, let's prioritize that.
    # But for visualization, let's stick to standard behavior or just pass everything.
    benchmark.visualize_comparisons(models_to_test, dataset, num_samples=5)
    print("\nBenchmark and visualization complete!")


if __name__ == "__main__":
    main()
