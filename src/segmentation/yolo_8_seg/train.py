# autopep8: off
import sys
import os
import shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

# Add project root to sys.path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from utils.utils import seed_everything, utils_cuda_clear
from segmentation.common.hparams import *
# Importing commons
#autopep8: on


# Path to the model checkpoint to resume. Set to None to start fresh.
RESUME_MODEL_PATH = r"/content/old/runs/segment/yolov8m_crack_seg/weights/last.pt"


def train_model(epochs=YOLO_EPOCHS, batch_size=YOLO_BATCH_SIZE, lr=YOLO_LEARNING_RATE, model_size=YOLO_MODEL_SIZE):
    """
    Main training wrapper for YOLOv8
    """
    # Initialize model
    resume_flag = False
    if RESUME_MODEL_PATH and os.path.exists(RESUME_MODEL_PATH):
        print(f"Resuming training from: {RESUME_MODEL_PATH}")
        model = YOLO(RESUME_MODEL_PATH)
        resume_flag = True
    else:
        if RESUME_MODEL_PATH:
            print(
                f"Warning: Checkpoint not found at {RESUME_MODEL_PATH}. Starting fresh.")

        model_name = f"yolov8{model_size}-seg.pt"  # e.g., yolov8n-seg.pt
        yaml_name = f"yolov8{model_size}-seg.yaml"

        if os.path.exists(model_name):
            print(f"Initializing YOLOv8 model from weights: {model_name}")
            model = YOLO(model_name)
        else:
            # Check for base config
            base_yaml = os.path.join(
                os.path.dirname(__file__), 'yolov8-seg.yaml')
            target_yaml = os.path.join(os.path.dirname(__file__), yaml_name)

            if os.path.exists(base_yaml):
                # Create scale-specific yaml if it doesn't exist
                if not os.path.exists(target_yaml):
                    print(
                        f"Creating {target_yaml} from {base_yaml} to apply scale '{model_size}'")
                    shutil.copy(base_yaml, target_yaml)

                print(f"Initializing YOLOv8 model from config: {target_yaml}")
                model = YOLO(target_yaml)
                # Note: This initializes with random weights!
                # If you want to load pretrained weights AND use custom config,
                # you might need to load weights then transfer.
                # But usually for 'best' custom arch, random init + pretrain is complex.
                # Standard ultralytics usage: if loading from yaml, it's new model.
                # To use pretrained weights with custom YAML is tricky if architecture changes.
                # simpler: YOLO(yaml).load(weights)
                try:
                    # Attempt to load matching pretrained weights if they exist online/locally
                    # This only works if architecture matches standard.
                    # If user customized yaml heavily, skip or use 'yolov8n-seg.pt' generic
                    weights = f"yolov8{model_size}-seg.pt"
                    print(
                        f"Attempting to load weights {weights} into custom config...")
                    model.load(weights)
                except Exception as e:
                    print(f"Could not load pretrained weights: {e}")
                    print(
                        "Starting from scratch (random weights) or incompatible config.")
            else:
                print(
                    f"Weights {model_name} not found and no config {base_yaml} found.")
                print(f"Initializing YOLOv8 model: {model_name} (downloading)")
                model = YOLO(model_name)

    # Dataset config path
    data_yaml = os.path.join(YOLO_DATASET_DIR, 'dataset.yaml')

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Dataset configuration not found at {data_yaml}. Run prepare.py first.")

    print(f"Starting training with:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Device: {DEVICE}")
    print(f"  Data: {data_yaml}")

    # TensorBoard Writer
    log_dir = UNET_MODEL_TRAIN_LOG_DIR + \
        str(datetime.now().strftime('%Y.%m.%d.%H_%M')) + "_yolo"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to {log_dir}")

    # Callback to log metrics at end of epoch
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch

        # Log Training Losses
        # trainer.loss_items is formatted usually [box, seg, cls] for segmentation?
        # Use trainer.label_loss_items to get names if possible, or mapping
        # YOLOv8 Seg: box_loss, seg_loss, cls_loss

        if hasattr(trainer, 'loss_names') and hasattr(trainer, 'loss_items'):
            for i, name in enumerate(trainer.loss_names):
                val = trainer.loss_items[i]
                writer.add_scalar(f'{name}/train', val, epoch)

        # Log Validation Metrics
        # trainer.metrics is a dict with keys like 'metrics/mAP50(M)'
        if hasattr(trainer, 'metrics'):
            for key, val in trainer.metrics.items():
                # Clean up key names for better U-Net style matching if desired
                # e.g., 'metrics/mAP50(M)' -> 'mAP50_Mask/val'
                clean_key = key.replace(
                    'metrics/', '').replace('(M)', '_Mask').replace('(B)', '_Box')
                writer.add_scalar(f'{clean_key}/val', val, epoch)

        # Learning Rate
        if hasattr(trainer, 'optimizer'):
            lr = trainer.optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', lr, epoch)

        # Flush
        writer.flush()

    def on_train_end(trainer):
        writer.close()

    # Register callbacks
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # Train
    # YOLOv8 handles logging and saving automatically to 'runs/segment/train'
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        device=0 if DEVICE == 'cuda' else 'cpu',
        project='runs/segment',
        name=f"yolov8{model_size}_crack_seg",
        exist_ok=True,  # Overwrite existing experiment with same name if exists, or False to increment
        pretrained=True,
        resume=resume_flag,
        imgsz=640,  # Increased to 640 for better small crack resolution
        patience=EARLY_STOPPING_PATIENCE,
        # Optimization for best segmentation quality
        retina_masks=True,   # High-resolution masks
        overlap_mask=True,   # Masks can overlap (useful for merging)
        box=5.0,             # Reduced from 7.5 to force focus on mask/segmentation
        cls=0.5,             # Cls loss gain (default 0.5)
        dfl=1.5,             # DFL loss gain (default 1.5)
        # We can increase box/dfl if we care more about shape accuracy
    )

    print("Training complete.")
    writer.close()
    return model


def main():
    seed_everything(SEED)
    print(f"Seeding with {SEED}")

    train_model()


if __name__ == "__main__":
    main()
    utils_cuda_clear()
