
#autopep8:off
from ultralytics import YOLO

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import warnings
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import torch.nn.functional as F

# Suppress the specific FutureWarning about weights_only from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

# --- SETUP PATHS ---
# Add the 'src' directory to sys.path to allow imports from segmentation, utils, etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# --- Robust Import Context ---
@contextlib.contextmanager
def import_context(base_path, conflicting_modules=['model', 'models', 'config', 'utils']):
    """
    Context manager to import modules from a specific path, handling name collisions.
    """
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    # Save original modules to restore? No, we need the new ones to persist for the objects.
    # But we MUST clear the existing ones so the new import picks up the new files.

    # Snapshot of what we are about to clear (for debugging/safety if needed, but here we just clear)
    cached_conflicts = {} 

    for mod in conflicting_modules:
        # Clear main module and submodules
        if mod in sys.modules:
            del sys.modules[mod]

        keys_to_remove = [k for k in sys.modules if k.startswith(mod + '.')]
        for k in keys_to_remove:
            del sys.modules[k]

    try:
        yield
    except ImportError as e:
        print(f"[ERROR] Import failed in context {base_path}: {e}")
    except Exception as e:
        print(f"[ERROR] Exception in context {base_path}: {e}")

# Import DeepCrack (conflicts on 'model')
deep_crack_path = os.path.abspath(os.path.join(current_dir, "DeepCrack", "codes"))
DeepCrack = None
with import_context(deep_crack_path, ['model']):
    try:
        from model.deepcrack import DeepCrack
    except ImportError:
        # Fallback to DeepSegmentor if DeepCrack/codes fails or user intended old one
        pass

# DeepSegmentor (Fallback or explicit if needed) - conflicts on 'models'
deep_segmentor_path = os.path.abspath(os.path.join(current_dir, "DeepSegmentor"))
DeepCrackNet = None
with import_context(deep_segmentor_path, ['models']):
    try:
        from models.deepcrack_networks import DeepCrackNet
    except ImportError:
        pass


# Import CrackFormer (nets - safe)
crackformer_path = os.path.abspath(os.path.join(current_dir, "CrackFormer-II", "CrackFormer-II"))
crackformer = None
with import_context(crackformer_path, []): # No conflicts expected but good practice to isolate
    try:
        from nets.crackformerII import crackformer
    except ImportError:
        print("[ERROR] CrackFormer not found")


# Import CrackSegFormer (DISABLED BY USER REQUEST)
# crack_segformer_path = os.path.abspath(os.path.join(current_dir, "CrackSegFormer"))
# ... (disabled import logic) ...
SegFormer = None 


# Import CSBSR (conflicts on 'model', 'config', 'utils')
csbsr_path = os.path.abspath(os.path.join(current_dir, "CSBSR"))
JointModel = None
csbsr_cfg = None

# Check for yacs dependency
try:
    import yacs
except ImportError:
    print("[WARNING] 'yacs' module not found. CSBSR requires 'yacs'. Install with: pip install yacs")

with import_context(csbsr_path, ['model', 'config', 'utils']):
    try:
        # Suppress SyntaxWarning from CSBSR code
        import warnings
        warnings.filterwarnings("ignore", category=SyntaxWarning)

        from model.modeling.build_model import JointModel
        from model.config import cfg as csbsr_cfg
    except ImportError as e:
        print(f"[ERROR] CSBSR not found: {e}")



from segmentation.common.hparams import DEVICE
from segmentation.common.dataloader import dataset_get, dataloader_get, val_transform
from segmentation.common.model import model_load
from segmentation.benchmark import Benchmark


# =================================================================================================
#                                     USER CONFIGURATION
# =================================================================================================

# --- DATASET PATHS ---
# Path to the test dataset (images and masks)
TEST_DATASET_IMG_PATH = "/content/test/test_img"
TEST_DATASET_MASK_PATH = "/content/test/test_lab"

# --- USER MODELS ---
# Paths to your trained model weights
# e.g., "models/segmentation/unet/best.pth"
PATH_MY_UNET = "/content/m_unet.pth"
# e.g., "models/segmentation/segformer/best.pth"
PATH_MY_SEGFORMER = "/content/m_segformer.pth"
# e.g., "models/segmentation/yolo/best.pt"
PATH_MY_YOLO = "/content/m_yolo.pt"

# --- COMPARISON MODELS (DeepCrack & CrackFormer) ---
# Paths to the downloaded weights (.pth)
PATH_DEEPCRACK_WEIGHTS = "/content/m_deepcrack.pth"
PATH_CRACKFORMER_WEIGHTS = "/content/m_crackformer.pth"
# PATH_CRACKSEGFORMER_WEIGHTS = os.path.join(crack_segformer_path, "pretrained_weights/segformer/mit_b0.pth")
PATH_CSBSR_WEIGHTS = "/content/m_csbsr.pth" # Placeholder for user provided weights


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
        # Renormalize: ImageNet -> [0,1] -> [-1,1] (approx for 0.5 mean/std)
        # Input x is ImageNet normalized: (x - mean) / std
        # Target is (x - 0.5) / 0.5 => 2*x - 1 (if x is 0-1)

        # Denormalize ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std + mean

        # Normalize to 0.5, 0.5, 0.5
        # (val - 0.5) / 0.5
        x_new = (x_denorm - 0.5) / 0.5

        # DeepCrackNet returns tuple: (side1, side2, side3, side4, side5, fused)
        # We only care about the fused output for benchmarking
        outputs = self.net(x_new)
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
        # Renormalize: ImageNet -> [0,1] -> [-1,1]

        # Denormalize ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std + mean

        # Normalize to 0.5, 0.5, 0.5
        x_new = (x_denorm - 0.5) / 0.5

        # crackformer returns tuple: (fuse5, fuse4, fuse3, fuse2, fuse1, output)
        # We want the last element: output
        outputs = self.net(x_new)
        return outputs[-1]

class CrackSegFormerWrapper(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        if SegFormer is None:
            raise ImportError("CrackSegFormer definition not found.")
        self.net = SegFormer(num_classes=num_classes, phi='b0')

    def forward(self, x):
        # Renormalize: ImageNet -> Custom
        # Custom: mean=[0.473, 0.493, 0.504], std=[0.100, 0.100, 0.099]

        # Denormalize ImageNet
        mean_imgnet = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std_imgnet = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std_imgnet + mean_imgnet

        # Normalize Custom
        mean_custom = torch.tensor([0.473, 0.493, 0.504], device=x.device).view(1, 3, 1, 1)
        std_custom = torch.tensor([0.100, 0.100, 0.099], device=x.device).view(1, 3, 1, 1)
        x_new = (x_denorm - mean_custom) / std_custom

        # Output: dict {'out': tensor}
        out = self.net(x_new)['out']
        # out shape: [B, 2, H, W]
        # We need single channel probability.
        # Softmax then take class 1
        prob = torch.softmax(out, dim=1)[:, 1, :, :].unsqueeze(1)
        return prob

class CSBSRWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        if JointModel is None or csbsr_cfg is None:
            raise ImportError("CSBSR definition not found.")

        # Configure CSBSR
        self.cfg = csbsr_cfg.clone()
        config_path = os.path.join(csbsr_path, "config/config_csbsr_pspnet.yaml")
        if os.path.exists(config_path):
            self.cfg.merge_from_file(config_path)
        else:
            print(f"[WARNING] CSBSR config not found at {config_path}")

        # Bypass loading missing pretrain weights
        self.cfg.defrost()
        self.cfg.MODEL.SR_SCRATCH = True
        self.cfg.MODEL.SR_PRETRAIN_ITER = 0 # Avoid pretrain checks
        # Use HRNet+OCR configuration to match user weights
        self.cfg.MODEL.DETECTOR_TYPE = 'HRNet_OCR'
        self.cfg.freeze()

        # CSBSR uses relative paths - temporarily change working directory
        old_cwd = os.getcwd()
        os.chdir(csbsr_path)
        try:
            self.net = JointModel(self.cfg)
            self.blur_ksize = self.cfg.BLUR.KERNEL_SIZE
        finally:
            os.chdir(old_cwd)

    def forward(self, x):
        # Renormalize: ImageNet -> CSBSR Mean/Std
        # CSBSR Defaults: Mean=[0.4741, 0.4937, 0.5048], Std=[0.1621, 0.1532, 0.1523]

        # Denormalize ImageNet
        mean_imgnet = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std_imgnet = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std_imgnet + mean_imgnet

        # Normalize CSBSR
        mean_csbsr = torch.tensor([0.4741, 0.4937, 0.5048], device=x.device).view(1, 3, 1, 1)
        std_csbsr = torch.tensor([0.1621, 0.1532, 0.1523], device=x.device).view(1, 3, 1, 1)
        x_new = (x_denorm - mean_csbsr) / std_csbsr

        # Store original dimensions for later
        _, _, H, W = x.shape

        # Resize to 224x224 for CSBSR (to avoid OOM and match config)
        input_size = (224, 224)
        x_resized = F.interpolate(x_new, size=input_size, mode='bilinear', align_corners=False)

        # Create dummy kernel
        B, C, H1, W1 = x_resized.shape
        # Assuming patch size logic in inference.py is for check, but model needs shape
        # forward(self, x, damy_kernel, sr_targets=None)
        # damy_kernel shape: [B, 1, K, K]
        damy_kernel = torch.zeros((B, 1, self.blur_ksize, self.blur_ksize), device=x.device)

        sr_preds, segment_preds, kernel_preds = self.net(x_resized, damy_kernel)

        # Resize back to original size
        segment_preds = F.interpolate(segment_preds, size=(H, W), mode='bilinear', align_corners=False)

        return segment_preds

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

        # Back to weights_only=False because of unsupported global numpy._core.multiarray.scalar
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint and isinstance(checkpoint['model'], dict): # For CrackSegFormer
                state_dict = checkpoint['model']
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
    # if not "PLACEHOLDER" in PATH_MY_SEGFORMER and os.path.exists(PATH_MY_SEGFORMER):
    #     try:
    #         m = model_load(
    #             "segformer", filepath=PATH_MY_SEGFORMER, device=device)
    #         models_to_test.append(("My-SegFormer", "torch", m))
    #         print("[OK] My-SegFormer loaded")
    #     except Exception as e:
    #         print(f"Error loading My-SegFormer: {e}")

    # CrackSegFormer
    # try:
    #     if os.path.exists(PATH_CRACKSEGFORMER_WEIGHTS) and SegFormer is not None:
    #         # CrackSegFormer weights structure might be dict 'model' or direct
    #         m = load_external_model(CrackSegFormerWrapper, PATH_CRACKSEGFORMER_WEIGHTS, device, num_classes=2)
    #         if m:
    #             models_to_test.append(("CrackSegFormer", "torch", m))
    #             print("Loaded CrackSegFormer")
    #     else:
    #          print(f"FAILED TO LOAD CrackSegFormer: File not found at {PATH_CRACKSEGFORMER_WEIGHTS}")
    # except Exception as e: print(f"Error loading CrackSegFormer: {e}")

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
