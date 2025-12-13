# autopep8: off
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
import random

# -------------------------importing common and utils -----------------------------

original_sys_path = sys.path.copy()

# moving to "src/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

# importing commons
from autoencoder.dataloader import *
from autoencoder.model import *
from autoencoder.hparams import *

# importing utils
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path

# --------------------------------------------------------------------------------


def load_model(model_path):
    model = model_init()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use the same transform as validation
    transform = get_val_transforms()
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0) # Add batch dim

    return image_tensor, image


def predict(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)
        return output


def visualize_reconstruction(original, reconstructed, diff_map, loss, save_path=None):
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert tensors to numpy
    if isinstance(original, torch.Tensor):
        original = original.squeeze().cpu().permute(1, 2, 0).numpy()
        original = original * std + mean
        original = np.clip(original, 0, 1)

    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.squeeze().cpu().permute(1, 2, 0).numpy()
        reconstructed = reconstructed * std + mean
        reconstructed = np.clip(reconstructed, 0, 1)

    if isinstance(diff_map, torch.Tensor):
        diff_map = diff_map.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Input")
    axes[0].axis('off')

    axes[1].imshow(reconstructed)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    # Heatmap for difference
    im = axes[2].imshow(diff_map, cmap='jet', vmin=0, vmax=diff_map.max())
    axes[2].set_title(f"Difference Map (MSE: {loss:.6f})")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")

    plt.show()


def main():
    model_path = f"{MODEL_DIR}/model.pth"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Find a random image from test set (try both Negative and Positive)
    # Assuming TEST_DIR points to kaggle-set root or similar
    # We want to test on both healthy and cracked to see the difference

    # Check if TEST_DIR has subfolders
    subdirs = glob(os.path.join(TEST_DIR, "*"))
    image_files = []

    if subdirs:
        # Recursive search
        image_files = glob(os.path.join(TEST_DIR, "**", "*.jpg"), recursive=True)
    else:
        image_files = glob(os.path.join(TEST_DIR, "*.jpg"))

    if not image_files:
        print(f"No images found in {TEST_DIR}")
        return

    # Pick a random image
    img_path = random.choice(image_files)
    img_path = "dog.jpg"

    print(f"Processing image: {img_path}")

    # Predict
    img_tensor, original_cv2 = preprocess_image(img_path)
    reconstructed = predict(model, img_tensor)

    # Compute difference map (MSE per pixel, averaged over channels)
    diff = (img_tensor.to(DEVICE) - reconstructed) ** 2
    diff_map = torch.mean(diff, dim=1) # Average over channels [1, H, W]
    loss = torch.mean(diff).item()

    print(f"Reconstruction MSE Loss: {loss:.6f}")

    visualize_reconstruction(img_tensor, reconstructed, diff_map, loss)


if __name__ == "__main__":
    main()
