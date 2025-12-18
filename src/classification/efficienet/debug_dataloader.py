
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path - ensuring we point to 'src' directory
# Current file is in src/classification/efficienet/
# We need to go up 3 levels: efficienet -> classification -> src
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# Now we can import from classification...
try:
    from classification.common.dataloader import dataloader_init
    from classification.common.hparams import *
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def show_batch():
    print("Initializing dataloader...")
    try:
        train_dl, _ = dataloader_init(batch_size=8)
    except Exception as e:
        print(f"Dataloader Init Error: {e}")
        return

    print("Fetching batch...")
    # Get a batch
    try:
        batch = next(iter(train_dl))
        images, labels = batch
    except Exception as e:
        print(f"Batch Fetch Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Labels unique: {torch.unique(labels)}")

    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    images_np = images.permute(0, 2, 3, 1).numpy()
    images_np = std * images_np + mean
    images_np = np.clip(images_np, 0, 1)

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < len(images_np):
            ax.imshow(images_np[idx])
            ax.set_title(f"Label: {labels[idx].item()}")
            ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(
        __file__), 'debug_dataloader_batch.png')
    plt.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    show_batch()
