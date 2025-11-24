# autopep8: off
import sys
import os

# Fix for Colab: Prevent TensorFlow from accessing GPU to avoid conflicts with PyTorch
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torchvision



from dataloader import *
from model import *
from hparams import *

original_sys_path = sys.path 
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

# importing utils
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path
# autopep8: on

# --------------------------------------------------------------------------------


def train_epoch(model, loader, criterion, optimizer, device, writer, epoch, step):
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, desc='Training', leave=False, position=0)
    for batch_idx, (images, targets) in enumerate(loop):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix({'loss': f'{loss.item():.6f}'})

        step += 1
        writer.add_scalar('batch/train_loss', loss.item(), step)

    epoch_loss = running_loss / len(loader)
    return epoch_loss, step


def validate(model, loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0

    # For visualization
    vis_images = []
    vis_outputs = []

    loop = tqdm(loader, desc='Validating', leave=False, position=0)
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loop):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.6f}'})

            # Save first batch for visualization
            if batch_idx == 0:
                vis_images = images[:8]
                vis_outputs = outputs[:8]

    epoch_loss = running_loss / len(loader)

    # Log images
    # Denormalize for visualization (approximate)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    vis_images = vis_images * std + mean
    vis_outputs = vis_outputs * std + mean

    # Clamp to 0-1
    vis_images = torch.clamp(vis_images, 0, 1)
    vis_outputs = torch.clamp(vis_outputs, 0, 1)

    grid_in = torchvision.utils.make_grid(vis_images, nrow=4)
    grid_out = torchvision.utils.make_grid(vis_outputs, nrow=4)

    writer.add_image('val/input_images', grid_in, epoch)
    writer.add_image('val/reconstructed_images', grid_out, epoch)

    return epoch_loss


def train_model():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = f"{LOG_DIR}/{timestamp}"
    writer = SummaryWriter(log_dir)

    seed_everything(SEED)

    train_loader, val_loader = dataloader_init()
    print("Dataloader initialized.")

    model = model_init().to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=SCHEDULER_PATIENCE
    )

    best_loss = float('inf')
    epochs_without_improvement = 0
    step = 0

    print("Starting training...")

    epoch_loop = tqdm(range(EPOCHS), desc='Epochs', leave=False, position=0)
    for epoch in epoch_loop:
        train_loss, step = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, writer, epoch, step
        )

        val_loss = validate(
            model, val_loader, criterion, DEVICE, writer, epoch
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = f"{MODEL_DIR}/autoencoder_best.pth"
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_save(model, best_model_path, epoch, optimizer, val_loss)
            epochs_without_improvement = 0
            epoch_loop.set_description(f'Epochs (Best Loss: {best_loss:.6f})')
        else:
            epochs_without_improvement += 1
            epoch_loop.set_description(
                f'Epochs (Best Loss: {best_loss:.6f}, No improv: {epochs_without_improvement})')

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Training completed.")
    writer.close()


if __name__ == "__main__":
    train_model()
    utils_cuda_clear()
