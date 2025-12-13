# autopep8: off
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch
from torch.amp import autocast, GradScaler
# -------------------------importing common and utils -----------------------------

original_sys_path = sys.path.copy()

# moving to "segmentation/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))
# importing commons
from segmentation.common.dataloader import *
from segmentation.common.loss_function import *
from segmentation.common.model import *
from segmentation.common.hparams import *
# importing utils
from utils.utils import *
#autopep8: on
# go back to the origin path
sys.path = original_sys_path

# --------------------------------------------------------------------------------

# -----------------metrics-----------------


def dice_metric(y_pred, y_true, smooth=1e-6):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    dice_score = (2. * intersection + smooth) / \
        (y_pred.sum() + y_true.sum() + smooth)
    return dice_score


def iou_metric(y_pred, y_true, smooth=1e-6):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    total = (y_pred + y_true).sum()
    union = total - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score
# -----------------------------------------


def calculate_metrics(predictions, targets, threshold=0.5):
    # predictions: [B, 1, H, W] or [B, H, W]
    # targets: [B, 1, H, W] or [B, H, W]

    # Binary prediction
    preds = (predictions > threshold).float()
    targets = targets.float()

    # flatten per image
    batch_size = preds.shape[0]
    preds_flat = preds.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)

    # True/False Positives/Negatives per image
    TP = ((preds_flat == 1) & (targets_flat == 1)).sum(dim=1).float()
    TN = ((preds_flat == 0) & (targets_flat == 0)).sum(dim=1).float()
    FP = ((preds_flat == 1) & (targets_flat == 0)).sum(dim=1).float()
    FN = ((preds_flat == 0) & (targets_flat == 1)).sum(dim=1).float()

    # dividing by zero counter measure
    epsilon = 1e-7

    # Per image metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    specificity = TN / (TN + FP + epsilon)

    # IoU (Intersection over Union)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    # if union = 0 (huge IOU, unlikely to happend, but needs to be handled)
    iou = intersection / (union + epsilon)
    iou[union == 0] = 1.0

    # Dice Coefficient
    dice_denom = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * intersection) / (dice_denom + epsilon)
    dice[dice_denom == 0] = 1.0

    # Confusion table (summed over batch for logging)
    conf_table = [[TP.sum().item(), FP.sum().item()], [
        FN.sum().item(), TN.sum().item()]]

    return {
        'accuracy': accuracy.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1_score': f1_score.mean().item(),
        'specificity': specificity.mean().item(),
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'confusion_table': conf_table
    }


def train_epoch(model, train_loader, criterion, optimizer, device, scaler):

    model.train()
    running_loss = .0
    metrics = {
        'iou': [], 'dice': [], 'recall': [],
        'precision': [], 'f1_score': []
    }

    loop = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda'):
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        with torch.no_grad():
            predictions_sigmoid = torch.sigmoid(predictions)
            batch_metrics = calculate_metrics(predictions_sigmoid, masks)

            for key in metrics.keys():
                value = batch_metrics[key]
                if isinstance(value, torch.Tensor):
                    value = value.cpu().item()
                metrics[key].append(batch_metrics[key])

        loop.set_postfix({'loss': loss.item()})

    loop.close()

    avg_loss = running_loss / len(train_loader)
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    metrics = {
        'iou': [], 'dice': [], 'recall': [],
        'precision': [], 'f1_score': []
    }

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            with autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions, masks)

            running_loss += loss.item()

            # Calculate metrics
            predictions_sigmoid = torch.sigmoid(predictions)
            batch_metrics = calculate_metrics(predictions_sigmoid, masks)

            for key in metrics.keys():
                value = batch_metrics[key]
                if isinstance(value, torch.Tensor):
                    value = value.cpu().item()
                metrics[key].append(batch_metrics[key])

    avg_loss = running_loss / len(val_loader)
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


def train_model(writer, epochs: int = UNET_EPOCHS, batch_size: int = UNET_BATCH_SIZE, lr: float = UNET_LEARNING_RATE, device=DEVICE):
    """
    Main training loop
    """

    train_dl, valid_dl = dataloader_init(batch_size=batch_size)
    print("Dataloader initialized.")

    device = torch.device(DEVICE)

    model = model_init(model_name="unet").to(device)

    criterion = DiceLoss()
    print("Criterion initialized: DiceLoss")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=UNET_WEIGHT_DECAY,
    )
    print(f"Optimizer initialized: Adam")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max iou
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-7
    )
    print(f"Scheduler initialized: ReduceLROnPlateau")

    scaler = GradScaler('cuda')

    # params for inter-epoch model accuracy checking
    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0

    writer.add_text("Hparams", f"""
    -           Learning Rate: {lr}
    -              Batch Size: {batch_size}
    -            Weight Decay: {UNET_WEIGHT_DECAY}
    -                  Epochs: {epochs}
    -      Scheduler Patience: {SCHEDULER_PATIENCE}
    - Early Stopping Patience: {EARLY_STOPPING_PATIENCE}
    -                  Device: {DEVICE}
    """)

    loop = tqdm(range(epochs), desc="Epochs", leave=False)
    for epoch in loop:

        train_metrics = train_epoch(
            model, train_dl, criterion, optimizer, device, scaler)
        val_metrics = validate(model, valid_dl, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['iou'])

        # Log metrics
        metrics_to_log = ['loss', 'iou', 'dice',
                          'precision', 'recall', 'f1_score']
        for metric in metrics_to_log:
            metric_name = metric.replace(
                '_', '-').title() if metric != 'iou' else 'IoU'
            writer.add_scalar(f'{metric_name}/train',
                              train_metrics[metric], epoch)
            writer.add_scalar(f'{metric_name}/val', val_metrics[metric], epoch)

        # Learning Rate
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # SAVE BEST MODEL
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }
            model_save(model, checkpoint,
                       filename=f"model_unet_{best_val_iou}.pth",
                       filepath=UNET_MODEL_TRAIN_DIR)
        else:
            loop.set_description(
                f'Epochs (Best IOU: {best_val_iou:.2f}%, No improvement: {patience_counter}/{EARLY_STOPPING_PATIENCE})')
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered!")
            print(f"   No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            print(f"   Best IoU: {best_val_iou:.4f} at epoch {best_epoch + 1}")
            break

    writer.close()
    print("Training done.")
    return model


def main() -> int:

    writer = SummaryWriter(UNET_MODEL_TRAIN_LOG_DIR +
                           str(datetime.now().strftime('%Y.%m.%d.%H_%M')))

    seed_everything(SEED)
    print(f"Seeding with {SEED}")

    model = train_model(writer=writer)


if __name__ == "__main__":
    main()
    utils_cuda_clear()
