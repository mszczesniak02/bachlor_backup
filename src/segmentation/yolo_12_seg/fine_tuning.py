
from utils.utils import seed_everything, utils_cuda_clear
from segmentation.common.hparams import *
import sys
import os
import itertools
from datetime import datetime
from ultralytics import YOLO

# Add project root to sys.path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# Importing commons


def tune_single_model(bsize, lr, epochs, device=DEVICE, model_size=YOLO_MODEL_SIZE):
    """
    Runs training for a single set of hyperparameters.
    """
    print(f"\n--- Tuning Run: Batch={bsize}, LR={lr} ---")

    # Initialize model
    model_name = f"yolo12{model_size}-seg.pt"
    model = YOLO(model_name)

    data_yaml = os.path.join(YOLO_DATASET_DIR, 'dataset.yaml')

    project_path = 'runs/segment/tuning'
    name = f"bs_{bsize}_lr_{lr:.1e}"

    # Train
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=bsize,
        lr0=lr,
        device=0 if device == 'cuda' else 'cpu',
        project=project_path,
        name=name,
        exist_ok=True,
        imgsz=512,
        plots=False,  # Reduce overhead
        save=False   # Don't save final models to save space during tuning, unless desired
    )

    # Get metrics
    # Ultralytics returns a metrics object after training, or we can read from results.csv
    # The 'train' method returns a Results object? No, it returns None or path usually.
    # Actually, model.train() returns None in some versions, but let's check validation metrics.

    metrics = model.val(split='val')

    # metrics.box.map    # map50-95
    # metrics.seg.map    # map50-95
    # metrics.seg.map50  # map50

    best_map = metrics.seg.map
    print(f"--- Result: mAP50-95 = {best_map:.4f} ---")

    return best_map


def run_tuning(batch_sizes, lrs, epochs=3):
    print("Tuning hyperParams")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Learning rates: {lrs}")

    results = {}

    for bsize, lr in itertools.product(batch_sizes, lrs):
        try:
            score = tune_single_model(bsize, lr, epochs)
            results[(bsize, lr)] = score
        except Exception as e:
            print(f"Run failed for bs={bsize}, lr={lr}: {e}")
            results[(bsize, lr)] = 0.0

    print("\nHyperparameter tuning completed!")
    print("Results:")
    best_params = None
    best_score = -1

    for (bs, lr), score in results.items():
        print(f"  Batch: {bs}, LR: {lr:.1e} -> mAP: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = (bs, lr)

    print(
        f"\nBest Params: Batch={best_params[0]}, LR={best_params[1]:.1e} (mAP={best_score:.4f})")


def main():
    seed_everything(SEED)

    # Tuning params
    epochs = 5  # Short epochs for tuning
    batch_sizes = [8, 16]
    lrs = [1e-2, 1e-3, 1e-4]

    run_tuning(batch_sizes, lrs, epochs)


if __name__ == "__main__":
    main()
    utils_cuda_clear()
