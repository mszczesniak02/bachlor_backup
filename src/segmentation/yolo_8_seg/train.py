# autopep8: off
import sys
import os
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


def train_model(epochs=YOLO_EPOCHS, batch_size=YOLO_BATCH_SIZE, lr=YOLO_LEARNING_RATE, model_size=YOLO_MODEL_SIZE):
    """
    Main training wrapper for YOLOv8
    """
    # Initialize model
    model_name = f"yolov8{model_size}-seg.pt"  # e.g., yolov8n-seg.pt
    print(f"Initializing YOLOv8 model: {model_name}")
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
        imgsz=512,  # Matching U-Net size (updated to 512 per user request)
        patience=EARLY_STOPPING_PATIENCE,
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
