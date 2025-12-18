
import torch
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2
import sys
import os
#autopep8:off

# 1. Fix sys.path BEFORE any other project imports
# current_file_path = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file_path)  # src/segmentation/yolo_12_seg
# src_path = os.path.abspath(os.path.join(current_dir, '../../'))  # src
# from segmentation.common.hparams import YOLO_DATASET_DIR, DEVICE

YOLO_DATASET_DIR =  r"../../../../datasets/yolo_seg_data"
DEVICE = "cpu"


# 2. Now safe to import project modules


def load_label_mask(label_path, img_shape):
    """
    Reads YOLO label file and returns a binary mask (merged polygons).
    img_shape: (height, width)
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        # parts[0] is class_id
        coords = parts[1:]

        # Reshape to (N, 2)
        points = np.array(coords).reshape(-1, 2)

        # Denormalize
        points[:, 0] *= w
        points[:, 1] *= h

        points = points.astype(np.int32)

        # Draw polygon
        cv2.fillPoly(mask, [points], 1)

    return mask


def get_prediction_mask(result, img_shape):
    """
    Extracts binary mask from YOLO result (merging all detections).
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return mask

    # YOLOv8/12 masks are often smaller size, need to resize or use result.masks.xy (polygons)
    # Using polygons (xy) is safer for resolution matching

    for poly in result.masks.xy:
        if len(poly) == 0:
            continue
        points = poly.astype(np.int32)
        cv2.fillPoly(mask, [points], 1)

    return mask


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def main():
    # 1. Model Selection
    # Trying to find the 'm' model as requested, or falling back
    possible_models = [
        # Local training output (relative to this script)
        os.path.abspath(os.path.join(os.path.dirname(__file__), "runs/segment/yolov12m_crack_seg/weights/best.pt")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "runs/segment/yolov12n_crack_seg/weights/best.pt")),

        # Original hardcoded paths (updated for yolo12 naming just in case)
        "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segmentation/yolo_big/runs/segment/yolov12m_crack_seg/weights/best.pt", 
        "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segmentation/yolo_big/runs/segment/yolov12n_crack_seg/weights/best.pt",
    ]

    model_path = None
    for p in possible_models:
        full_path = p
        if not os.path.isabs(p):
            full_path = os.path.join(os.path.dirname(__file__), p)

        if os.path.exists(full_path):
            model_path = full_path
            break

    if model_path is None:
        print("Warning: Could not find trained model (checked for 'm' and 'n'). Using 'yolo12m-seg.pt' generic.")
        model_path = "yolo12m-seg.pt"

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # 2. Dataset Setup
    val_img_dir = os.path.join(YOLO_DATASET_DIR, 'images', 'val')
    val_lbl_dir = os.path.join(YOLO_DATASET_DIR, 'labels', 'val')

    if not os.path.exists(val_img_dir):
        print(f"Error: Validation images not found at {val_img_dir}")
        return

    image_files = sorted([f for f in os.listdir(
        val_img_dir) if f.endswith(('.jpg', '.png'))])

    ious = []

    print(f"Starting validation on {len(image_files)} images...")

    for img_file in tqdm(image_files):
        img_path = os.path.join(val_img_dir, img_file)

        # Load image to get shape
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 1. Get Ground Truth Mask
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(val_lbl_dir, lbl_file)

        gt_mask = load_label_mask(lbl_path, (h, w))

        # 2. Get Prediction
        # Force 512 size as per common config, verify if this matches training
        results = model.predict(img_path, conf=0.25, verbose=False, imgsz=512)
        pred_mask = get_prediction_mask(results[0], (h, w))

        # 3. Calculate IoU
        iou = calculate_iou(gt_mask, pred_mask)
        ious.append(iou)

        # VISUALIZATION: Save 3 examples of images with MULTIPLE predictions to see variety
        if not hasattr(main, "demo_count"):
            main.demo_count = 0

        if main.demo_count < 3 and results[0].masks is not None and len(results[0].masks) > 1:
            main.demo_count += 1

            # 1. Individual Detections (Multi-color)
            vis_individual = img.copy()
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

            for i, poly in enumerate(results[0].masks.xy):
                if len(poly) == 0: continue
                points = poly.astype(np.int32)
                color = colors[i % len(colors)]
                cv2.drawContours(vis_individual, [points], -1, color, 2)

            # 2. Merged Prediction (Binary -> White on Black)
            vis_merged = (pred_mask * 255).astype(np.uint8)
            vis_merged_bgr = cv2.cvtColor(vis_merged, cv2.COLOR_GRAY2BGR)

            # Add text headers
            cv2.putText(vis_individual, f"Individual Preds ({len(results[0].masks)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_merged_bgr, "Merged Binary Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Concatenate
            combined_multi = np.hstack((vis_individual, vis_merged_bgr))

            output_vis_multi = f"merged_prediction_multicolor_{main.demo_count}.jpg"
            cv2.imwrite(output_vis_multi, combined_multi)
            print(f"\n[INFO] Saved multi-color merging comparison to {output_vis_multi}")


    mean_iou = np.mean(ious)
    print("\n" + "="*30)
    print(f"Validation Results")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Images: {len(image_files)}")
    print(f"Mean IoU (Merged Class): {mean_iou:.4f}")
    print("="*30)


if __name__ == "__main__":
    main()
