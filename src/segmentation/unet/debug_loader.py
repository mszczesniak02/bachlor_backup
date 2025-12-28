
#autopep8: off
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Add project root to path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from segmentation.common.hparams import *
from segmentation.common.dataloader import dataloader_init, IMG_TRAIN_PATH

def denormalize(tensor):
    """
    Reverse the normalization for visualization.
    Mean: (0.485, 0.456, 0.406), Std: (0.229, 0.224, 0.225)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def main():
    print(f"Checking dataset at: {IMG_TRAIN_PATH}")

    try:
        train_dl, valid_dl = dataloader_init(batch_size=4)
    except Exception as e:
        print(f"Failed to init dataloader: {e}")
        return

    print("Fetching one batch...")
    try:
        images, masks = next(iter(train_dl))
    except Exception as e:
        print(f"Failed to fetch batch: {e}")
        return

    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Mask unique values: {torch.unique(masks)}")
    print(f"Mask mean value: {masks.float().mean().item()}")

    # Denormalize images
    images_denorm = torch.stack([denormalize(img) for img in images])

    # Create a grid
    # Images
    img_grid = make_grid(images_denorm, nrow=4)
    img_np = img_grid.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    # Masks
    msk_grid = make_grid(masks, nrow=4)
    # (H, W, 1) usually but make_grid makes it (3, H, W)
    msk_np = msk_grid.permute(1, 2, 0).numpy()

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.imshow(img_np)
    plt.title("Batch Images (Denormalized)")
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(msk_np[:, :, 0], cmap='gray')  # Take first channel
    plt.title("Batch Masks")
    plt.axis('off')

    output_file = "debug_dataloader_batch.png"
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")

    if masks.max() == 0:
        print("WARNING: Masks are completely empty (all zeros)!")
    elif masks.max() > 1:
        print("WARNING: Masks contain values > 1 (not normalized to 0-1)!")
    else:
        print("Masks look essentially correct (0-1 range).")


if __name__ == "__main__":
    main()
