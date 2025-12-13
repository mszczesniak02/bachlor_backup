
# autopep8: off
import sys
import os
import itertools
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

    for images, labels in loader:
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

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def run_tuning():
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    # Hyperparameters to tune
    learning_rates = [1e-3, 1e-4, 5e-5]
    batch_sizes = [16, 32]

    # Generate all combinations
    combinations = list(itertools.product(learning_rates, batch_sizes))
    print(f"Total combinations to test: {len(combinations)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    base_log_dir = f"{LOG_DIR}/{timestamp}_tuning"

    results = []

    for i, (lr, batch_size) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: LR={lr}, Batch={batch_size}")

        # Init components
        seed_everything(SEED)
        train_dl, val_dl, _ = dataloader_init(batch_size=batch_size)
        model = model_init(num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=ENTRY_WEIGHT_DECAY)
        scaler = GradScaler('cuda')

        run_name = f"lr_{lr}_bs_{batch_size}"
        writer = SummaryWriter(f"{base_log_dir}/{run_name}")

        best_acc = 0.0
        patience_counter = 0

        # Training loop (shorter for tuning)
        epochs = 15
        loop = tqdm(range(epochs), desc=f"Tuning {run_name}", leave=False)

        for epoch in loop:
            train_loss, train_acc = train_epoch(
                model, train_dl, criterion, optimizer, device, scaler)
            val_loss, val_acc = validate(model, val_dl, criterion, device)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 5:  # Stricter patience for tuning
                break

        writer.add_hparams(
            {'lr': lr, 'batch_size': batch_size},
            {'hparam/best_acc': best_acc}
        )
        writer.close()

        print(f"Result: Best Val Acc = {best_acc:.2f}%")
        results.append((lr, batch_size, best_acc))

        # Clean up
        del model, optimizer, scaler
        utils_cuda_clear()

    print("\n" + "="*50)
    print("TOP 3 COMBO RESULTS:")
    results.sort(key=lambda x: x[2], reverse=True)
    for res in results[:3]:
        print(f"LR: {res[0]}, Batch: {res[1]} -> Acc: {res[2]:.2f}%")
    print("="*50)


if __name__ == "__main__":
    run_tuning()
