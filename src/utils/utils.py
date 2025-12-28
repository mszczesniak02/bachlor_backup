from torch.cuda import empty_cache
import io
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2

import torch
import random
import os
import gc


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def utils_cuda_clear():
    """
    Clear the leftover memory from COLAB training model
    """

    print("Clearning memory...", end="")
    empty_cache()
    gc.collect()
    empty_cache()
    print("done.")


def plot_effect(image, mask, effect=[], effect_title="Transform"):
    """
    Plots 2 images if no effect is passed.
    """
    if len(effect) != 0:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Dane wejściowe")
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Maska")
        ax[2].imshow(effect, cmap="gray")
        ax[2].set_title(effect_title)
    else:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Mask")

    plt.show()


def np2ten(img: np.array) -> torch.tensor:
    """
    Numpy array to torch tensor
    Used for inference and cleanup
    """
    if len(img.shape) == 2:
        # Jeśli grayscale, dodaj wymiar kanału
        img = np.expand_dims(img, axis=-1)

        # Konwersja: [H, W, C] -> [C, H, W] i normalizacja do [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Dodaj wymiar batcha [1, C, H, W] - potrzebne dla modelu
        tensor = tensor.unsqueeze(0)

        return tensor


def ten2np(img: torch.Tensor, denormalize: bool = False) -> np.array:
    """
    Torch tensor to numpy array
    Used for inference and cleanup
    """
    # Usuń batch dimension jeśli istnieje
    if len(img.shape) == 4:
        img = img.squeeze(0)  # [1, C, H, W] -> [C, H, W]

    # Przenieś na CPU
    img = img.detach().cpu()

    # Denormalizacja jeśli wymagana (przed permutacją, bo mean/std są dla kanałów)
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

    # Konwersja: [C, H, W] -> [H, W, C] i na numpy
    img = img.permute(1, 2, 0).numpy()

    # Denormalizacja: [0, 1] -> [0, 255]
    img = (img * 255).astype(np.uint8)

    # Jeśli grayscale (1 kanał), usuń wymiar kanału
    if img.shape[2] == 1:
        img = img.squeeze(2)

    return img


def visualize_model_output(image, mask_gt, mask_pred, save_path=None):
    """
    Visualizes original image, ground truth mask, predicted probability map (heatmap), and overlay.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 2. Ground Truth
    axes[1].imshow(mask_gt, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    # 3. Prediction Heatmap
    im3 = axes[2].imshow(mask_pred, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("Prediction Heatmap")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Overlay
    # Create binary mask for overlay from prediction
    binary_pred = mask_pred > 0.5
    overlay = image.copy()
    overlay[binary_pred] = [255, 0, 0]  # Red overlay

    # Alpha blend
    alpha = 0.4
    overlay_vis = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)

    axes[3].imshow(overlay_vis)
    axes[3].set_title("Overlay (Pred > 0.5)")
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")

    plt.show()
