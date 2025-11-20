# my own stuff
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch


from hparams import *
from model import *
from loss_function import *
from dataloader import *
import sys
import os
import torch

from utils import *


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


def train_epoch(model, train_loader, criterion, optimizer, device):

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
        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, masks)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def train_model(writer, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, device=DEVICE):
    """
    Main training loop
    """

    train_dl, valid_dl = dataloader_init(batch_size=BATCH_SIZE)
    print("Dataloader initialized.")

    device = torch.device(DEVICE)

    model = model_init().to(device)
    criterion = DiceLoss()
    print("Criterion initialized: DiceLoss")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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

    # params for inter-epoch model accuracy checking
    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0

    writer.add_text("Hparams", f"""
    -           Learning Rate: {LEARNING_RATE}
    -              Batch Size: {BATCH_SIZE}
    -            Weight Decay: {WEIGHT_DECAY}
    -                  Epochs: {EPOCHS}
    -      Scheduler Patience: {SCHEDULER_PATIENCE}
    - Early Stopping Patience: {EARLY_STOPPING_PATIENCE}
    -                  Device: {DEVICE}
    """)

    loop = tqdm(range(EPOCHS), desc="Epochs", leave=False)
    for epoch in loop:

        train_metrics = train_epoch(
            model, train_dl, criterion, optimizer, device)
        val_metrics = validate(model, valid_dl, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['iou'])

        # Loss
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)

        # IoU
        writer.add_scalars('IoU', {
            'train': train_metrics['iou'],
            'val': val_metrics['iou']
        }, epoch)

        # Dice
        writer.add_scalars('Dice', {
            'train': train_metrics['dice'],
            'val': val_metrics['dice']
        }, epoch)

        # Precision
        writer.add_scalars('Precision', {
            'train': train_metrics['precision'],
            'val': val_metrics['precision']
        }, epoch)

        # Recall
        writer.add_scalars('Recall', {
            'train': train_metrics['recall'],
            'val': val_metrics['recall']
        }, epoch)

        # F1-Score
        writer.add_scalars('F1-Score', {
            'train': train_metrics['f1_score'],
            'val': val_metrics['f1_score']
        }, epoch)

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
            model_save(model, checkpoint, filename=f"model_{best_val_iou}.pth")
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

    writer = SummaryWriter(MODEL_TRAIN_LOG_DIR +
                           str(datetime.now().strftime('H%M')))

    seed_everything(SEED)
    print(f"Seeding with {SEED}")

    model = train_model(writer=writer)


if __name__ == "__main__":
    main()
    utils_cuda_clear()
