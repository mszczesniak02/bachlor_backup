# autopep8: off
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch
from torch.amp import autocast, GradScaler
import numpy as np

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

# go back to the origin path
sys.path = original_sys_path

# --------------------------------------------------------------------------------



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
        for images, masks in tqdm(val_loader, desc="Validation", leave=False):

            images = images.to(device)
            masks = masks.to(device)

            with autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions, masks)

            running_loss += loss.item()

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


def tune_single_model(bsize, lr, epochs, global_pbar, device=DEVICE):
    """
    Runs training for a single set of hyperparameters.
    Returns best_val_iou and best_epoch.
    """

    run_name = f"bs_{bsize}_lr_{lr:.1e}"
    # Writer per run
    writer = SummaryWriter(SEGFORMER_MODEL_TRAIN_LOG_DIR + "/tuning_" + datetime.now().strftime('%H%M') + "/" + run_name)

    # Re-init dataloaders for new batch size
    train_dl, val_dl = dataloader_init(batch_size=bsize)

    model = model_init(model_name="segformer")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=SEGFORMER_WEIGHT_DECAY,
    )
    criterion = DiceLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-7
    )

    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0

    scaler = GradScaler('cuda')

    # No internal tqdm for epochs, we update global_pbar
    for epoch in range(epochs):
        # Update global description
        global_pbar.set_description(f"Run: {run_name} | Epoch {epoch+1}/{epochs} | Best IoU: {best_val_iou:.4f}")

        train_metrics = train_epoch(
            model, train_dl, criterion, optimizer, device, scaler)
        val_metrics = validate(model, val_dl, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['iou'])

        # Log metrics to tensorboard (PER EPOCH only)
        # This keeps logs small
        metrics_to_log = ['loss', 'iou', 'dice', 'precision', 'recall', 'f1_score']
        for metric in metrics_to_log:
            metric_name = metric.replace('_', '-').title() if metric != 'iou' else 'IoU'
            writer.add_scalar(f'{metric_name}/train', train_metrics[metric], epoch)
            writer.add_scalar(f'{metric_name}/val', val_metrics[metric], epoch)

        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Track best
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Update global progress bar
        global_pbar.update(1)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            # If early stopping triggers, we need to update the pbar for the skipped epochs
            # so the total count remains correct
            skipped_epochs = epochs - (epoch + 1)
            if skipped_epochs > 0:
                global_pbar.update(skipped_epochs)
            break

    # Log hparams summary for this run
    hparam_dict = {
        'batch_size': bsize,
        'learning_rate': lr,
        'weight_decay': SEGFORMER_WEIGHT_DECAY,
    }

    metric_dict = {
        'hparam/best_val_iou': best_val_iou,
        'hparam/best_epoch': best_epoch,
    }

    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    return best_val_iou, best_epoch


def run_tuning(batch_sizes, lrs, epochs=3):
    print("Tuning hyperParams")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Learning rates: {lrs}")

    total_epochs = len(batch_sizes) * len(lrs) * epochs

    # Global progress bar
    with tqdm(total=total_epochs, desc="Total Tuning Progress") as global_pbar:
        for bsize in batch_sizes:
            for lr in lrs:
                tune_single_model(bsize, lr, epochs, global_pbar)

    print("\nHyperparameter tuning completed!")


def main():
    # tuning params
    epochs = 3
    batch_sizes = [4, 8]
    lrs = [1e-4, 6e-5, 1e-5]

    run_tuning(batch_sizes, lrs, epochs)


if __name__ == "__main__":
    main()
    utils_cuda_clear()
