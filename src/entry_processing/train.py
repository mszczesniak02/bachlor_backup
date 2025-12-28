
# autopep8: off
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# -------------------------importing common and utils -----------------------------

original_sys_path = sys.path.copy()

# moving to "src/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

from entry_processing.dataloader import *
from entry_processing.model import *
from entry_processing.hparams import *
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path
# autopep8: on

# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
RESUME_CHECKPOINT = None  # "path/to/checkpoint.pth"


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc='Training', leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        acc = (predicted == labels).float().mean().item()
        loop.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(loader)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return epoch_loss, metrics, all_labels, all_preds


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc='Validating', leave=False)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            acc = (predicted == labels).float().mean().item()
            loop.set_postfix(
                {'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(loader)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return epoch_loss, metrics, all_labels, all_preds


def train_model():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = f"{LOG_DIR}/{timestamp}" if 'LOG_DIR' in globals(
    ) else f"runs/entry_processing/{timestamp}"
    writer = SummaryWriter(log_dir)

    print(f"Device: {DEVICE}")
    device = torch.device(DEVICE)
    seed_everything(SEED)

    train_dl, val_dl, test_dl = dataloader_init(batch_size=ENTRY_BATCH_SIZE)
    print("Dataloaders initialized.")

    model = model_init(num_classes=NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ENTRY_LEARNING_RATE, weight_decay=ENTRY_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=ENTRY_SCHEDULER_PATIENCE, verbose=True
    )

    scaler = GradScaler('cuda')

    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0

    start_epoch = 0
    if RESUME_CHECKPOINT is not None and os.path.isfile(RESUME_CHECKPOINT):
        print(f"Loading checkpoint from {RESUME_CHECKPOINT}")
        checkpoint = torch.load(
            RESUME_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

    print("Starting training...")

    epochs = ENTRY_EPOCHS
    loop = tqdm(range(start_epoch, epochs), desc='Epochs')

    class_names = ["no_crack", "crack"]

    for epoch in loop:
        train_loss, train_metrics, train_labels, train_preds = train_epoch(
            model, train_dl, criterion, optimizer, device, scaler)
        val_loss, val_metrics, val_labels, val_preds = validate(
            model, val_dl, criterion, device)

        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            metric_name = metric.replace('_', '-').title()
            writer.add_scalar(f'{metric_name}/train',
                              train_metrics[metric], epoch)
            writer.add_scalar(f'{metric_name}/val', val_metrics[metric], epoch)

        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Confusion Matrix
        cm_train = confusion_matrix(train_labels, train_preds)
        cm_val = confusion_matrix(val_labels, val_preds)

        fig_train = plot_confusion_matrix(cm_train, class_names)
        writer.add_figure('train/confusion_matrix', fig_train, epoch)

        fig_val = plot_confusion_matrix(cm_val, class_names)
        writer.add_figure('val/confusion_matrix', fig_val, epoch)

        scheduler.step(val_loss)

        # for compatibility with old logic using percentage
        val_acc = val_metrics['accuracy'] * 100
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            patience_counter = 0

            save_path = f"{MODEL_DIR}/best_model.pth"
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(
                save_path) else '.', exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)

            loop.set_description(f"Epochs (Best Acc: {best_acc:.2f}%)")
        else:
            patience_counter += 1
            loop.set_description(
                f"Epochs (Best Acc: {best_acc:.2f}%, No improv: {patience_counter}/{ENTRY_PATIENCE})")

        if patience_counter >= ENTRY_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("Training finished.")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    writer.close()

    # Optional: Run on Test Set
    print("\nRunning on Test Set with Best Model...")
    checkpoint = torch.load(f"{MODEL_DIR}/best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics, _, _ = validate(model, test_dl, criterion, device)
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")


if __name__ == "__main__":
    train_model()
    utils_cuda_clear()
