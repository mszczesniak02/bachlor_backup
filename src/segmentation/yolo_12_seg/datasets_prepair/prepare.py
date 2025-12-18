#autopep8: off
import os
import shutil
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import sys

# Add project root to sys.path to import hparams
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../../')))

from src.segmentation.common.hparams import *
#autopep8: on


def create_dir_structure(base_path):
    """Creates the standard YOLOv8 directory structure."""
    dirs = [
        os.path.join(base_path, 'images', 'train'),
        os.path.join(base_path, 'images', 'val'),
        os.path.join(base_path, 'labels', 'train'),
        os.path.join(base_path, 'labels', 'val')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def mask_to_yolo_polygon(mask, normalize=True):
    """
    Converts a binary mask to YOLO polygon format.
    Args:
        mask: Binary mask numpy array.
        normalize: Whether to normalize coordinates to [0, 1].
    Returns:
        List of polygon strings in format "class_id x1 y1 x2 y2 ...".
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []

    for contour in contours:
        # Filter small contours to avoid noise
        if cv2.contourArea(contour) < 50:
            continue

        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        coords = approx.flatten().astype(float)

        if normalize:
            coords[0::2] /= w  # x coordinates
            coords[1::2] /= h  # y coordinates

        # Limit to [0, 1]
        coords = np.clip(coords, 0, 1)

        # Format: class_id x1 y1 x2 y2 ...
        # Assuming single class "0" for now
        poly_str = "0 " + " ".join([f"{x:.6f}" for x in coords])
        polygons.append(poly_str)

    return polygons


def process_dataset(img_dir, mask_dir, output_img_dir, output_label_dir):
    """
    Processes images and masks, converting them to YOLO format.
    """
    print(f"Processing data from {img_dir} to {output_img_dir}...")

    valid_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
    image_files = sorted([f for f in os.listdir(
        img_dir) if f.lower().endswith(valid_extensions)])

    for img_file in tqdm(image_files):
        img_path = os.path.join(img_dir, img_file)
        # Assuming mask has same name as image
        mask_path = os.path.join(mask_dir, img_file)

        # Determine strict mask name match or check extensions if names differ slightly
        if not os.path.exists(mask_path):
            # Try to find corresponding mask with different extension if exact match fails
            base_name = os.path.splitext(img_file)[0]
            found = False
            for ext in valid_extensions:
                potential_mask = os.path.join(mask_dir, base_name + ext)
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    found = True
                    break
            if not found:
                print(f"Warning: Mask for {img_file} not found. Skipping.")
                continue

        # Copy image
        shutil.copy2(img_path, os.path.join(output_img_dir, img_file))

        # Process mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Binarize
        mask = (mask > 127).astype(np.uint8) * 255

        polygons = mask_to_yolo_polygon(mask)

        # Write labels
        label_file = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(output_label_dir, label_file), 'w') as f:
            for poly in polygons:
                f.write(poly + "\n")


def create_dataset_yaml(output_dir):
    """Creates the dataset.yaml file required by YOLOv8."""
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'crack'
        }
    }

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(
        f"Created dataset.yaml at {os.path.join(output_dir, 'dataset.yaml')}")


def main():
    # Paths from hparams
    # Resolve relative paths in hparams relative to THIS script location logic if needed,
    # but hparams usually has absolute or relative-to-root paths.
    # Note: hparams.py provided paths like "../../../../datasets/..." which are relative to hparams.py location potentially.
    # Let's trust python import to resolve constants, but we need to ensure we run this script from correct context or handle paths.

    # We will assume hparams paths are relative to the project root or are valid absolute paths.
    # If they are relative like "../../../", we need to be careful.

    # Let's make paths absolute based on hparams location if they are relative
    # The hparams file is in src/segmentation/common/

    # src/segmentation/yolo_8_seg/datasets_prepair
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))

    # Helper to resolve path
    def resolve_path(path):
        if os.path.isabs(path):
            return path
        # Assume path is relative to src/segmentation/common/hparams.py or project root?
        # looking at hparams, it uses "../../../../datasets", which implies it is relative to src/segmentation/common/
        # so let's resolve it relative to src/segmentation/common
        hparams_dir = os.path.join(
            project_root, 'src', 'segmentation', 'common')
        return os.path.abspath(os.path.join(hparams_dir, path))

    train_img_dir = resolve_path(IMG_TRAIN_PATH)
    train_mask_dir = resolve_path(MASK_TRAIN_PATH)
    val_img_dir = resolve_path(IMG_TEST_PATH)
    val_mask_dir = resolve_path(MASK_TEST_PATH)
    output_dir = resolve_path(YOLO_DATASET_DIR)

    print(f"Dataset Prep Configuration:")
    print(f"  Train Images: {train_img_dir}")
    print(f"  Train Masks:  {train_mask_dir}")
    print(f"  Val Images:   {val_img_dir}")
    print(f"  Val Masks:    {val_mask_dir}")
    print(f"  Output Dir:   {output_dir}")

    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Cleaning up...")
        shutil.rmtree(output_dir)

    img_train_out, img_val_out, lbl_train_out, lbl_val_out = create_dir_structure(
        output_dir)

    process_dataset(train_img_dir, train_mask_dir,
                    img_train_out, lbl_train_out)
    process_dataset(val_img_dir, val_mask_dir, img_val_out, lbl_val_out)

    create_dataset_yaml(output_dir)
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
