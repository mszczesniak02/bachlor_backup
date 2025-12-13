
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


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        loop.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc='Validating', leave=False)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            acc = 100. * correct / total
            loop.set_postfix(
                {'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    # Use LOG_DIR from hparams or define local default
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

    print("Starting training...")

    epochs = ENTRY_EPOCHS
    loop = tqdm(range(epochs), desc='Epochs')

    for epoch in loop:
        train_loss, train_acc = train_epoch(
            model, train_dl, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_dl, criterion, device)

        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        # Save best model logic (based on lowest loss usually better for classification stability, or highest acc)
        # Using accuracy here as primary metric for simplicity, or loss if desired.
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
    checkpoint = torch.load(f"{MODEL_DIR}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_dl, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    train_model()
    utils_cuda_clear()
