# autopep8: off
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from dataloader import *
from model import *
from hparams import *

original_sys_path = sys.path.copy()

# moving to "src/" (assuming we are in src/autoencoder)
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))
# importing utils
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path

# --------------------------------------------------------------------------------


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (images, targets) in enumerate(loop):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validation", leave=False)
        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            running_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = running_loss / len(val_loader)
    return avg_loss


def tune_single_model(bsize, lr, epochs, global_pbar, device=DEVICE):
    """
    Runs training for a single set of hyperparameters.
    Returns best_val_loss and best_epoch.
    """

    run_name = f"bs_{bsize}_lr_{lr:.1e}"
    # Writer per run
    # Using LOG_DIR from hparams if available, else default
    log_dir = LOG_DIR if 'LOG_DIR' in globals() else "../../../models_log/autoencoder/"
    writer = SummaryWriter(log_dir + "/tuning_" + datetime.now().strftime('%H%M') + "/" + run_name)

    # Re-init dataloaders for new batch size
    train_dl, val_dl = dataloader_init(batch_size=bsize)

    model = model_init()
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-7
    )

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    # Use a smaller patience for tuning to save time
    TUNING_PATIENCE = 5

    # No internal tqdm for epochs, we update global_pbar
    for epoch in range(epochs):
        # Update global description
        global_pbar.set_description(f"Run: {run_name} | Epoch {epoch+1}/{epochs} | Best Loss: {best_val_loss:.6f}")

        train_loss = train_epoch(
            model, train_dl, criterion, optimizer, device)
        val_loss = validate(model, val_dl, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Update global progress bar
        global_pbar.update(1)

        if patience_counter >= TUNING_PATIENCE:
            # If early stopping triggers, we need to update the pbar for the skipped epochs
            skipped_epochs = epochs - (epoch + 1)
            if skipped_epochs > 0:
                global_pbar.update(skipped_epochs)
            break

    # Log hparams summary for this run
    hparam_dict = {
        'batch_size': bsize,
        'learning_rate': lr,
        'weight_decay': WEIGHT_DECAY,
    }

    metric_dict = {
        'hparam/best_val_loss': best_val_loss,
        'hparam/best_epoch': best_epoch,
    }

    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    return best_val_loss, best_epoch


def run_tuning(batch_sizes, lrs, epochs=5):
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
    epochs = 5
    batch_sizes = [8, 16, 32]
    lrs = [1e-3, 5e-4, 1e-4]

    run_tuning(batch_sizes, lrs, epochs)


if __name__ == "__main__":
    main()
    utils_cuda_clear()
