
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

# Configuration
DATASET_PATH = "/home/krzeslaav/Projects/datasets/dataset_segmentation"
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "train_img")
TRAIN_LAB_DIR = os.path.join(DATASET_PATH, "train_lab")


def visualize_dataset_samples():
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(TRAIN_LAB_DIR):
        print(f"Error: Dataset directories not found at {DATASET_PATH}")
        return

    # Get list of all images
    all_images = list(Path(TRAIN_IMG_DIR).glob("*"))
    valid_images = [img for img in all_images if img.suffix.lower() in [
        '.jpg', '.png', '.jpeg', '.bmp']]

    if not valid_images:
        print("No images found.")
        return

    # Find valid pairs (image + label)
    valid_pairs = []

    # Shuffle to pick random candidates efficiently
    random.shuffle(valid_images)

    for img_path in valid_images:
        # Assuming label has same basename + .png (standard for segmentation masks)
        lab_path = Path(TRAIN_LAB_DIR) / (img_path.stem + ".png")

        if not lab_path.exists():
            # Try original extension if distinct format not found
            lab_path = Path(TRAIN_LAB_DIR) / img_path.name

        if lab_path.exists():
            valid_pairs.append((img_path, lab_path))

        if len(valid_pairs) >= 10:
            break

    if len(valid_pairs) < 10:
        print(f"Not enough pairs found. Found only {len(valid_pairs)}")
        return

    # Setup plots
    fig_imgs, axes_imgs = plt.subplots(2, 5, figsize=(20, 8))
    fig_masks, axes_masks = plt.subplots(2, 5, figsize=(20, 8))

    fig_imgs.suptitle("Training Data Samples: Images", fontsize=16)
    fig_masks.suptitle("Training Data Samples: Masks", fontsize=16)

    axes_imgs = axes_imgs.flatten()
    axes_masks = axes_masks.flatten()

    for i, (img_path, lab_path) in enumerate(valid_pairs):
        # Load Image
        try:
            img = Image.open(img_path)
            mask = Image.open(lab_path)

            # Plot Image
            axes_imgs[i].imshow(img)
            axes_imgs[i].axis('off')
            # axes_imgs[i].set_title(img_path.name, fontsize=8)

            # Plot Mask
            # Convert mask to numpy to check values if needed, usually they are 0/255 or 0/1
            # displaying with 'gray' cmap covers both
            axes_masks[i].imshow(mask, cmap='gray')
            axes_masks[i].axis('off')
            # axes_masks[i].set_title(lab_path.name, fontsize=8)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_dataset_samples()
