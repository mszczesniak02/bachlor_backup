#autopep8: off
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn as nn

from ultralytics import YOLO
import segmentation_models_pytorch as smp               # preset model
import torchvision.models as models

#hparams
from torch import cuda
import os
SEGFORMER_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segformermodel_segformer_0.5864474233337809.pth"
UNET_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/model_unet_0.5960555357910763.pth"
YOLO_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segmentation/yolo_big/runs/segment/yolov8m_crack_seg/weights/best.pt"
EFFICIENTNET_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/classification/efficientnet/model_f1_0.9171_epoch14.pth"
CONVNEXT_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/classification/convnext/model_f1_0.9258_epoch9.pth"
DOMAIN_CONTROLLER_PATH = r"/home/krzeslaav/Projects/bachlor/models/entry_classificator/best_model.pth"
# --- Config ---
OUTPUT_IMAGE_SIZE = 512
DEVICE = "cpu" if cuda.is_available() else "cpu"
NUM_CLASSES = 4
IMAGE_PATH_0 = r"/home/krzeslaav/Projects/bachlor/image_test_0.jpg"
IMAGE_PATH_1 = r"/home/krzeslaav/Projects/bachlor/image_test_1.jpg"

# --- Dataset ---
SEG_IMG_TEST_PATH = r"/home/krzeslaav/Projects/datasets/dataset_segmentation/test_img"
SEG_MASK_TEST_PATH = r"/home/krzeslaav/Projects/datasets/dataset_segmentation/test_lab"

CLASS_IMG_TEST_PATH_ROOT = r"/home/krzeslaav/Projects/datasets/classification_width/test_img"


# hpaams end

# from hparams import *

import matplotlib.pyplot as plt


def load_segformer(filepath=SEGFORMER_PATH, device=DEVICE):
    model = smp.Segformer(
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    if os.path.isfile(filepath):
        checkpoint = torch.load(
            filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        return model
    else:
        print(f"ERROR: segformer {filepath} not found, returning empty model")
        return model


def load_unet(filepath=UNET_PATH, device=DEVICE):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    if os.path.isfile(filepath):
        checkpoint = torch.load(
            filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    return model


def load_yolo(filepath=YOLO_PATH, device=DEVICE):
    model = YOLO(filepath)
    return model


def load_efficientnet(filepath=EFFICIENTNET_PATH, device=DEVICE):

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )

    if os.path.isfile(filepath):
        checkpoint = torch.load(
            filepath, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model


def load_convnext(filepath=CONVNEXT_PATH, device=DEVICE):

    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)

    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, NUM_CLASSES)

    if os.path.isfile(filepath):
        checkpoint = torch.load(
            filepath, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model


def load_domain_controller(filepath=DOMAIN_CONTROLLER_PATH, device=DEVICE):
    original_sys_path = sys.path.copy()
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../')))

    from entry_processing.model import model_load
    sys.path = original_sys_path

    model = model_load(filepath, device)
    return model


def load_all_models(device=DEVICE):
    model_segformer = load_segformer(device=device)
    model_unet = load_unet(device=device)
    model_yolo = load_yolo(device=device)
    model_efficientnet = load_efficientnet(device=device)
    model_convnext = load_convnext(device=device)
    model_domain_controller = load_domain_controller(device=device)
    print(f"All models loaded on {device}.")
    return model_segformer, model_unet, model_yolo, model_efficientnet, model_convnext, model_domain_controller


def load_image(filepath):
    img_cv2 = cv2.imread(filepath)
    if img_cv2 is None:
        raise ValueError(f"Image not found: {filepath}")
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_cv2, (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))

    # Tensor
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(
        2, 0, 1).float().unsqueeze(0)

    return img_tensor, img_resized


def visualize_prediction(img_numpy, final_mask, binary_mask, category, masks_dict, weights_dict):
    """
    Visualizes the prediction results including individual model outputs and ensemble weights.
    """
    plt.figure(figsize=(20, 10))

    # Row 1: Pipeline Results
    plt.subplot(2, 4, 1)
    plt.imshow(img_numpy)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(final_mask, cmap='jet', vmin=0, vmax=1)
    plt.title(f"Ensemble Heatmap")
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')

    plt.subplot(2, 4, 4)
    if np.sum(binary_mask) > 0:
        masked_vis = (img_numpy * binary_mask[:, :, None]).astype(np.uint8)
        plt.imshow(masked_vis)
    else:
        plt.text(0.5, 0.5, "No Mask", ha='center', va='center')
    plt.title(f"Class: {category}")
    plt.axis('off')

    # Row 2: Individual Models
    plot_idx = 5
    for name, mask in masks_dict.items():
        if mask is not None:
            if plot_idx <= 8:
                plt.subplot(2, 4, plot_idx)
                # Use jet colormap for individual models too, to match ensemble style
                plt.imshow(mask, cmap='jet', vmin=0, vmax=1)
                weight = weights_dict.get(name, 0.0)
                plt.title(f"{name} (w={weight:.2f})")
                plt.axis('off')
                plot_idx += 1

    plt.tight_layout()
    plt.show()


def predict(filepath, model_segformer, model_unet, model_yolo, model_efficientnet, model_convnext, model_domain_controller):
    img_tensor, img_numpy = load_image(filepath)
    img_tensor = img_tensor.to(DEVICE)

    # 1. Domain Control
    if model_domain_controller is not None:
        try:
            model_domain_controller.eval()
            with torch.no_grad():
                reconstruction = model_domain_controller(img_tensor)
                mse_loss = F.mse_loss(reconstruction, img_tensor).item()
                print(f"Domain MSE: {mse_loss:.6f}")
        except Exception as e:
            print(f"Domain check error: {e}")

    # 2. Segmentation Ensemble
    weights = {
        "unet": 1.0,
        "segformer": 1.0,
        "yolo": 1.0
    }

    active_weights_sum = 0
    if model_unet:
        active_weights_sum += weights["unet"]
    if model_segformer:
        active_weights_sum += weights["segformer"]
    if model_yolo:
        active_weights_sum += weights["yolo"]

    normalized_weights = {}
    if active_weights_sum > 0:
        if model_unet:
            normalized_weights["unet"] = weights["unet"] / active_weights_sum
        if model_segformer:
            normalized_weights["segformer"] = weights["segformer"] / \
                active_weights_sum
        if model_yolo:
            normalized_weights["yolo"] = weights["yolo"] / active_weights_sum

    print(f"Ensemble Weights: {normalized_weights}")

    mask_accum = np.zeros(
        (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.float32)
    masks_dict = {}

    # U-Net
    if model_unet:
        try:
            model_unet.eval()
            with torch.no_grad():
                out = model_unet(img_tensor)
                mask_unet = torch.sigmoid(out).squeeze().cpu().numpy()
                mask_accum += mask_unet * normalized_weights["unet"]
                masks_dict["unet"] = mask_unet
        except Exception as e:
            print(f"U-Net inference error: {e}")

    # SegFormer
    if model_segformer:
        try:
            model_segformer.eval()
            with torch.no_grad():
                out = model_segformer(img_tensor)
                out = F.interpolate(out, size=(
                    OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), mode='bilinear', align_corners=False)
                mask_segformer = torch.sigmoid(out).squeeze().cpu().numpy()
                mask_accum += mask_segformer * normalized_weights["segformer"]
                masks_dict["segformer"] = mask_segformer
        except Exception as e:
            print(f"SegFormer inference error: {e}")

    # YOLO
    if model_yolo:
        try:
            results = model_yolo.predict(
                img_numpy, imgsz=OUTPUT_IMAGE_SIZE, verbose=False, device=0 if DEVICE == 'cuda' else 'cpu')
            if results and results[0].masks is not None:
                masks = results[0].masks.data
                if masks.shape[1:] != (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE):
                    masks = F.interpolate(masks.unsqueeze(1), size=(
                        OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(1)
                mask_yolo = torch.any(masks > 0.5, dim=0).float().cpu().numpy()
                mask_accum += mask_yolo * normalized_weights["yolo"]
                masks_dict["yolo"] = mask_yolo
            else:
                masked_null = np.zeros(
                    (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.float32)
                masks_dict["yolo"] = masked_null
        except Exception as e:
            print(f"YOLO inference error: {e}")

    final_mask = mask_accum
    binary_mask = (final_mask > 0.5).astype(np.float32)

    # 3. Classification
    category = "Unknown"
    if np.sum(binary_mask) == 0:
        category = "No Crack Detected (Empty Mask)"

    if model_efficientnet and np.sum(binary_mask) > 0:
        try:
            mask_tensor = torch.from_numpy(binary_mask).to(
                DEVICE).unsqueeze(0).unsqueeze(0)
            masked_img = img_tensor * mask_tensor

            model_efficientnet.eval()
            with torch.no_grad():
                out = model_efficientnet(masked_img)
                if model_convnext:
                    model_convnext.eval()
                    out_conv = model_convnext(masked_img)
                    out = (out + out_conv) / 2

                _, pred = torch.max(out, 1)
                category = pred.item()
                print(f"Predicted Category: {category}")
        except Exception as e:
            print(f"Classification error: {e}")

    # Visualize
    visualize_prediction(img_numpy, final_mask, binary_mask,
                         category, masks_dict, normalized_weights)

    # Save outputs
    output_dir = "output_predictions"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # Save individual masks
    for name, mask in masks_dict.items():
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            save_path = os.path.join(output_dir, f"{base_name}_{name}.png")
            cv2.imwrite(save_path, mask_uint8)
            print(f"Saved {name} mask to {save_path}")

    # Save ensemble result
    final_mask_uint8 = (final_mask * 255).astype(np.uint8)
    save_path_ensemble = os.path.join(output_dir, f"{base_name}_ensemble.png")
    cv2.imwrite(save_path_ensemble, final_mask_uint8)
    print(f"Saved ensemble mask to {save_path_ensemble}")

    # Save binary ensemble result
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    save_path_binary = os.path.join(
        output_dir, f"{base_name}_ensemble_binary.png")
    cv2.imwrite(save_path_binary, binary_mask_uint8)
    print(f"Saved binary ensemble mask to {save_path_binary}")


if __name__ == "__main__":
    model_segformer, model_unet, model_yolo, model_efficientnet, model_convnext, model_domain_controller = load_all_models()
    predict(IMAGE_PATH_0, model_segformer, model_unet, model_yolo,
            model_efficientnet, model_convnext, model_domain_controller)
