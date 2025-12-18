#autopep8:off
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm
import cv2
import numpy as np
import time
from utils.utils import seed_everything
from segmentation.common.hparams import *
# autopep8:on


def evaluate_init(model_path):
    """Load model for inference."""
    model = YOLO(model_path)
    return model


def predict_single(model, img_path, device='cpu'):
    """Run inference on a single image."""
    results = model.predict(
        source=img_path,
        conf=0.25,
        device=0 if device == 'cuda' else 'cpu',
        save=False,
        imgsz=512
    )
    return results[0]


def visualize_result(img_path, result):
    """Display original image and prediction."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res_plotted = result.plot()
    res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis('off')

    ax[1].imshow(res_plotted)
    ax[1].set_title("Prediction")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


def bulk_inference(model, img_dir, amount=100):
    """Run inference on a folder of images and measure time."""
    image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(
        img_dir) if f.endswith(('.jpg', '.png'))])

    times = []

    for i, img_path in enumerate(tqdm(image_files[:amount], desc="Inference")):
        start_time = time.time()
        _ = model.predict(img_path, verbose=False,
                          device=0 if DEVICE == 'cuda' else 'cpu', imgsz=512)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times) * 1000
    print(f"\nInference Speed (avg): {avg_time:.2f} ms per image")
    return times


def main():
    # Example usage
    # Assume we have a trained model, or use a pretrained one for demo
    # For now, let's look for the best model from training if available, else default

    model_path = "yolov8n-seg.pt"  # Default to pretrained if no local model
    possible_model = os.path.join(
        'runs/segment/yolov8n_crack_seg/weights/best.pt')
    if os.path.exists(possible_model):
        model_path = possible_model
        print(f"Using trained model: {model_path}")

    model = evaluate_init(model_path)

    # 1. Bulk Inference Timing
    print("Running Bulk Inference Timing...")
    # Use validation images
    # We need raw images, they are in YOLO_DATASET_DIR/images/val

    val_images_dir = os.path.join(YOLO_DATASET_DIR, 'images', 'val')
    if os.path.exists(val_images_dir):
        bulk_inference(model, val_images_dir, amount=50)
    else:
        print(
            f"Validation directory {val_images_dir} not found. Skipping bulk.")

    # 2. Single Visualization (Interactive or save)
    # Just take first image from val
    if os.path.exists(val_images_dir):
        files = os.listdir(val_images_dir)
        if files:
            target_img = os.path.join(val_images_dir, files[0])
            print(f"Visualizing {target_img}...")
            res = predict_single(model, target_img, device=DEVICE)
            # visualize_result(target_img, res) # Uncomment if running in notebook/gui

            # Save plot
            res_plotted = res.plot()
            cv2.imwrite("inference_result.jpg", res_plotted)
            print("Saved inference_result.jpg")


if __name__ == "__main__":
    main()
