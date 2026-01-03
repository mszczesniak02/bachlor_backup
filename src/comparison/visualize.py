#autopep8:off

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import warnings
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import contextlib
import traceback

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*weights_only.*")

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
from segmentation.benchmark import ten2np
from segmentation.common.model import model_load
from segmentation.common.dataloader import dataset_get, val_transform
from segmentation.common.hparams import DEVICE
# --- Robust Import Context ---
@contextlib.contextmanager
def import_context(base_path, conflicting_modules=['model', 'models', 'config', 'utils']):
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    for mod in conflicting_modules:
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

# Import DeepCrack
deep_crack_path = os.path.abspath(os.path.join(current_dir, "DeepCrack", "codes"))
DeepCrack = None
with import_context(deep_crack_path, ['model']):
    try:
        from model.deepcrack import DeepCrack
    except ImportError: pass

# Import DeepSegmentor (DeepCrackNet)
deep_segmentor_path = os.path.abspath(os.path.join(current_dir, "DeepSegmentor"))
DeepCrackNet = None
with import_context(deep_segmentor_path, ['models']):
    try:
        from models.deepcrack_networks import DeepCrackNet
    except ImportError: pass


# Import CrackFormer
crackformer_path = os.path.abspath(os.path.join(current_dir, "CrackFormer-II", "CrackFormer-II"))
crackformer = None
with import_context(crackformer_path, []):
    try:
        from nets.crackformerII import crackformer
    except ImportError:
        print("[ERROR] CrackFormer not found")


# Import CrackSegFormer (DISABLED)
SegFormer = None
# ... disabled imports ...


# Import CSBSR
csbsr_path = os.path.abspath(os.path.join(current_dir, "CSBSR"))
JointModel = None
csbsr_cfg = None

# Check yacs
try:
    import yacs
except ImportError:
    print("[WARNING] 'yacs' module not found. CSBSR requires 'yacs'.")

with import_context(csbsr_path, ['model', 'config', 'utils']):
    try:
        import warnings
        warnings.filterwarnings("ignore", category=SyntaxWarning)

        from model.modeling.build_model import JointModel
        from model.config import cfg as csbsr_cfg
    except ImportError as e:
        print(f"[ERROR] CSBSR not found: {e}")


# --- CONFIGURATION (Match main.py) ---
TEST_DATASET_IMG_PATH = "/content/test/test_img"
TEST_DATASET_MASK_PATH = "/content/test/test_lab"

PATH_MY_UNET = "/content/m_unet.pth"
PATH_MY_SEGFORMER = "/content/m_segformer.pth"
PATH_MY_YOLO = "/content/m_yolo.pt"
PATH_DEEPCRACK_WEIGHTS = "/content/m_deepcrack.pth"
PATH_CRACKFORMER_WEIGHTS = "/content/m_crackformer.pth"
# PATH_CRACKSEGFORMER_WEIGHTS = os.path.join(crack_segformer_path, "pretrained_weights/segformer/mit_b0.pth")
PATH_CSBSR_WEIGHTS = "/content/m_csbsr.pth"


# --- WRAPPERS ---
class DeepCrackWrapper(nn.Module):
    def __init__(self, in_nc=3, num_classes=1, ngf=64, norm='batch'):
        super().__init__()
        if DeepCrackNet is None:
            raise ImportError("DeepCrackNet definition not found.")
        self.net = DeepCrackNet(in_nc, num_classes, ngf, norm)

    def forward(self, x):
        # Renormalize: ImageNet -> [0,1] -> [-1,1]

        # Denormalize ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std + mean

        # Normalize to 0.5, 0.5, 0.5
        x_new = (x_denorm - 0.5) / 0.5

        outputs = self.net(x_new)
        return outputs[-1] # fused


class CrackFormerWrapper(nn.Module):
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

        outputs = self.net(x_new)
        return outputs[-1] # output

class CrackSegFormerWrapper(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        if SegFormer is None:
            raise ImportError("CrackSegFormer definition not found.")
        self.net = SegFormer(num_classes=num_classes, phi='b0')

    def forward(self, x):
        # Renormalize: ImageNet -> Custom

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
        # Force U-Net16 configuration to match user weights
        self.cfg.MODEL.DETECTOR_TYPE = 'u-net16'
        self.cfg.MODEL.UP_SAMPLE_METHOD = 'interpolate' 
        self.cfg.freeze()

        self.net = JointModel(self.cfg)
        self.blur_ksize = self.cfg.BLUR.KERNEL_SIZE

    def forward(self, x):
        # Renormalize: ImageNet -> CSBSR Mean/Std

        # Denormalize ImageNet
        mean_imgnet = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std_imgnet = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std_imgnet + mean_imgnet

        # Normalize CSBSR
        mean_csbsr = torch.tensor([0.4741, 0.4937, 0.5048], device=x.device).view(1, 3, 1, 1)
        std_csbsr = torch.tensor([0.1621, 0.1532, 0.1523], device=x.device).view(1, 3, 1, 1)
        x_new = (x_denorm - mean_csbsr) / std_csbsr

        # Resize to 224x224 for CSBSR
        input_size = (224, 224)
        B, C, H, W = x.shape # Store original dimensions
        x_resized = F.interpolate(x_new, size=input_size, mode='bilinear', align_corners=False)

        # Create dummy kernel
        _, _, H1, W1 = x_resized.shape # Use resized dimensions for kernel
        damy_kernel = torch.zeros((B, 1, self.blur_ksize, self.blur_ksize), device=x.device)

        sr_preds, segment_preds, kernel_preds = self.net(x_resized, damy_kernel)

        # Resize back to original size
        segment_preds = F.interpolate(segment_preds, size=(H, W), mode='bilinear', align_corners=False)

        return segment_preds

# --- HELPER FUNCTIONS ---
def load_external_model(model_wrapper_class, weights_path, device, **kwargs):
    if not os.path.exists(weights_path):
        print(f"[WARN] File not found: {weights_path}")
        return None
    try:
        model = model_wrapper_class(**kwargs)
        checkpoint = torch.load(
            weights_path, map_location=device, weights_only=False)
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint




        # Clean keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '').replace(
                'net.', '').replace('netG.', '')
            if k.startswith('net.'):
                new_k = k[4:]  # handle wrapper prefixes if any
            new_state_dict[new_k] = v

        # Try loading
        missing, unexpected = model.load_state_dict(
            new_state_dict, strict=False)
        if hasattr(model, 'net'):
            # If wrapper, try loading into inner net
            missing, unexpected = model.net.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"[WARN] Partial load for {weights_path}. Missing {len(missing)} keys.")
                print(f"Sample missing: {missing[:5]}")
            else:
                print(f"[SUCCESS] Fully loaded weights for {weights_path} into model.net")
        else:
            if missing:
                print(f"[WARN] Partial load for {weights_path}. Missing {len(missing)} keys.")
                print(f"Sample missing: {missing[:5]}")
            else:
                print(f"[SUCCESS] Fully loaded weights for {weights_path}")

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[ERROR] Loading {weights_path}:")
        traceback.print_exc()
        return None


def predict_torch(model, images):
    outputs = model(images)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    probs = torch.sigmoid(outputs)
    return probs.cpu().detach().numpy().squeeze()


def predict_yolo(model, images_tensor):
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=images_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=images_tensor.device).view(1, 3, 1, 1)
    imgs_denorm = torch.clamp(images_tensor * std + mean, 0, 1)

    results = model.predict(imgs_denorm, imgsz=512, verbose=False, conf=0.25)

    batch_masks = []
    h, w = images_tensor.shape[2], images_tensor.shape[3]
    for res in results:
        if res.masks is not None:
            data = res.masks.data
            if data.shape[1:] != (h, w):
                data = torch.nn.functional.interpolate(
                    data.unsqueeze(1), size=(h, w), mode='bilinear').squeeze(1)
            # take max confidence as prob-ish
            mask_pred = torch.max(data, dim=0)[0]
        else:
            mask_pred = torch.zeros((h, w), device=images_tensor.device)
        batch_masks.append(mask_pred)

    if batch_masks:
        return torch.stack(batch_masks).cpu().numpy().squeeze()
    return np.zeros((h, w))


# --- MAIN VISUALIZATION ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running visualization on: {device}")

    # 1. Load Dataset
    print("Loading dataset...")
    try:
        dataset = dataset_get(img_path=TEST_DATASET_IMG_PATH,
                              mask_path=TEST_DATASET_MASK_PATH, transform=val_transform)
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load Models
    models = {}

    # User Models
    if os.path.exists(PATH_MY_UNET):
        models['My-UNet'] = model_load("unet",
                                       filepath=PATH_MY_UNET, device=device)
    else:
        print("My-UNet not found")

    if os.path.exists(PATH_MY_SEGFORMER):
        models['My-SegFormer'] = model_load("segformer",
                                            filepath=PATH_MY_SEGFORMER, device=device)
    else:
        print("My-SegFormer not found")

    if os.path.exists(PATH_MY_YOLO):
        models['My-YOLOv8'] = YOLO(PATH_MY_YOLO)
    else:
        print("My-YOLOv8 not found")

    # Comparison Models
    dc_model = load_external_model(
        DeepCrackWrapper, PATH_DEEPCRACK_WEIGHTS, device)
    if dc_model:
        models['DeepCrack'] = dc_model

    cf_model = load_external_model(CrackFormerWrapper, PATH_CRACKFORMER_WEIGHTS, device)
    if cf_model: models['CrackFormer'] = cf_model

    # csf_model = load_external_model(CrackSegFormerWrapper, PATH_CRACKSEGFORMER_WEIGHTS, device, num_classes=2)
    # if csf_model: models['CrackSegFormer'] = csf_model

    csbsr_model = load_external_model(CSBSRWrapper, PATH_CSBSR_WEIGHTS, device)
    if csbsr_model: models['CSBSR'] = csbsr_model

    print(f"Loaded models: {list(models.keys())}")

    # 3. Visualization Loop
    num_samples = 5
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    print(f"Generating {num_samples} visualizations...")

    for idx in indices:
        img_tensor, mask_tensor = dataset[idx]
        images = img_tensor.unsqueeze(0).to(device)

        preds = {}

        # Inference
        for name, model in models.items():
            with torch.no_grad():
                if name == 'My-YOLOv8':
                    pred = predict_yolo(model, images)
                else:  # Torch models
                    pred = predict_torch(model, images)
                preds[name] = pred

        # Ensemble (User Models Only)
        user_ensemble_keys = ['My-UNet', 'My-SegFormer', 'My-YOLOv8']
        available_keys = [k for k in user_ensemble_keys if k in preds]

        if available_keys:
            probs = [preds[k] for k in available_keys]
            ens_prob = np.mean(np.array(probs), axis=0)
            preds['Ensemble'] = ens_prob
        else:
            preds['Ensemble'] = np.zeros_like(
                preds.get('DeepCrack', next(iter(preds.values()))))  # Fallback

        # Plotting
        # Columns: Input, GT, UNet, SegFormer, YOLO8, DeepCrack, CrackFormer, CSBSR, Ensemble
        plot_order = ['Input', 'Ground Truth', 'My-UNet', 'My-SegFormer', 'My-YOLOv8', 'DeepCrack', 'CrackFormer', 'CSBSR', 'Ensemble']

        fig, ax = plt.subplots(1, 9, figsize=(36, 4))

        # Prepare content
        content = {}
        content['Input'] = ten2np(img_tensor, denormalize=True)
        content['Ground Truth'] = ten2np(mask_tensor, denormalize=False)
        content.update(preds)

        for i, key in enumerate(plot_order):
            ax[i].axis('off')
            ax[i].set_title(key)

            if key in content:
                img = content[key]
                if key == 'Input':
                    ax[i].imshow(img)
                else:
                    # Binarize for display unless it's prob
                    # Showing Probability map or binarized? Usually binarized for comparisons
                    # But soft edges are nice. Let's threshold at 0.5 for crisp comparison
                    if key != 'Ground Truth':
                        img = (img > 0.5).astype(np.float32)
                    ax[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                ax[i].text(0.5, 0.5, "Missing", ha='center', va='center')

        plt.tight_layout()
        out_file = f"comparative_visualization_{idx}.png"
        plt.savefig(out_file)
        plt.close()
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
