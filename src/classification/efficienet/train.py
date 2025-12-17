# autopep8: off
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# -------------------------importing common and utils -----------------------------

original_sys_path = sys.path.copy()

# moving to "classification/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# importing commons
from classification.common.dataloader import *
from classification.common.model import *
from classification.common.hparams import *

# importing utils
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path
# autopep8: on

# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
RESUME_CHECKPOINT = None  # "path/to/checkpoint.pth"


def plot_confusion_matrix(all_labels, all_preds, class_names, writer, epoch, tag='confusion_matrix'):
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Confusion Matrix - Epoch {epoch}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)

    return cm


def calculate_metrics(all_labels, all_preds, num_classes=5):
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), average='macro', zero_division=0
    )

    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), average=None, zero_division=0
    )

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }


def train_epoch(model, loader, criterion, optimizer, device, writer, epoch, step, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        acc = (predicted == labels).float().mean().item()
        loop.set_postfix(
            {'loss': f'{loss.item():.4f}', 'acc': f'{acc*100:.2f}%'})

        step += 1
        writer.add_scalar('batch/train_loss', loss.item(), step)
        writer.add_scalar('batch/train_acc', acc, step)

    epoch_loss = running_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds, num_classes=NUM_CLASSES)

    return epoch_loss, metrics, step, all_labels, all_preds


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
                {'loss': f'{loss.item():.4f}', 'acc': f'{acc*100:.2f}%'})

    epoch_loss = running_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds, num_classes=NUM_CLASSES)

    return epoch_loss, metrics, all_labels, all_preds


def train_model(writer, epochs=ENET_EPOCHS, batch_size=ENET_BATCH_SIZE, lr=ENET_LEARNING_RATE, device=DEVICE):
    class_names = ["1_wlosowe", "2_male", "3_srednie", "4_duze"]

    train_loader, val_loader = dataloader_init(batch_size=batch_size)
    print("Dataloader initialized.")

    # --- Class Weights Calculation ---
    print("Calculating class weights options...")
    class_dist = train_loader.dataset.get_class_distribution()
    total_samples = sum(class_dist.values())
    unique_classes = sorted(class_dist.keys())
    num_classes = len(unique_classes)

    print(f"Class distribution: {class_dist}")

    class_weights = []
    for i in range(num_classes):
        cnt = class_dist.get(i, 0)
        if cnt > 0:
            # weight = total / (classes * count)
            w = total_samples / (num_classes * cnt)
        else:
            w = 1.0
        class_weights.append(w)

    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Using class weights: {class_weights}")

    model = model_init(model_name="efficienet")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=ENET_WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=ENET_SCHEDULER_PATIENCE)

    hparams = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'image_size': ENET_IMAGE_SIZE,
        'weight_decay': ENET_WEIGHT_DECAY,
        'epochs': epochs,
        'patience': PATIENCE,
        'scheduler_patience': ENET_SCHEDULER_PATIENCE,
        'device': str(device),
        'model': 'EfficientNet-B0',
        'optimizer': 'AdamW',
        'loss_function': 'CrossEntropyLoss'
    }

    best_f1 = 0.0
    best_model_path = None
    best_model_path = None
    epochs_without_improvement = 0
    step = 0

    start_epoch = 0
    if RESUME_CHECKPOINT is not None and os.path.isfile(RESUME_CHECKPOINT):
        print(f"Loading checkpoint from {RESUME_CHECKPOINT}")
        checkpoint = torch.load(
            RESUME_CHECKPOINT, map_location=device, weights_only=False)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()

        # Filter keys with mismatching shapes
        filtered_state_dict = {k: v for k, v in state_dict.items(
        ) if k in model_state_dict and v.shape == model_state_dict[k].shape}

        if len(filtered_state_dict) != len(state_dict):
            print(
                f"Warning: {len(state_dict) - len(filtered_state_dict)} layers were skipped due to shape mismatch (likely classifier changes).")

        model.load_state_dict(filtered_state_dict, strict=False)

    scaler = GradScaler('cuda')

    print("Starting training...")
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    epoch_loop = tqdm(range(start_epoch, epochs), desc='Epochs', leave=False)
    for epoch in epoch_loop:
        train_loss, train_metrics, step, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch, step, scaler
        )

        val_loss, val_metrics, val_labels, val_preds = validate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']

        plot_confusion_matrix(train_labels, train_preds,
                              class_names, writer, epoch, 'train/confusion_matrix')
        plot_confusion_matrix(val_labels, val_preds, class_names,
                              writer, epoch, 'val/confusion_matrix')

        # Log metrics
        metrics_to_log = ['accuracy', 'f1_score', 'precision', 'recall']

        # Log Loss separately as it's not in the metrics dict in the same way (train_loss is a float, metrics is a dict)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        for metric in metrics_to_log:
            metric_name = metric.replace('_', '-').title()
            writer.add_scalar(f'{metric_name}/train',
                              train_metrics[metric], epoch)
            writer.add_scalar(f'{metric_name}/val', val_metrics[metric], epoch)

        # Per-class metrics
        for class_idx, class_name in enumerate(class_names):
            writer.add_scalar(
                f'Per_Class/f1_{class_name}', val_metrics['f1_per_class'][class_idx], epoch)
            writer.add_scalar(
                f'Per_Class/precision_{class_name}', val_metrics['precision_per_class'][class_idx], epoch)
            writer.add_scalar(
                f'Per_Class/recall_{class_name}', val_metrics['recall_per_class'][class_idx], epoch)

        writer.add_scalar('Learning_Rate', current_lr, epoch)

        scheduler.step(val_metrics['f1_score'])

        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            best_model_path = f"{ENET_MODEL_TRAIN_DIR}/model_f1_{best_f1:.4f}_epoch{epoch}.pth"
            os.makedirs(os.path.dirname(best_model_path) if os.path.dirname(
                best_model_path) else '.', exist_ok=True)
            model_save(model, best_model_path, epoch, optimizer, val_loss)
            epochs_without_improvement = 0
            epoch_loop.set_description(f'Epochs (Best F1: {best_f1:.4f})')
        else:
            epochs_without_improvement += 1
            epoch_loop.set_description(
                f'Epochs (Best F1: {best_f1:.4f}, No improv: {epochs_without_improvement}/{PATIENCE})')

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    if best_model_path:
        print(f"Best model saved: {best_model_path}")
    print("="*70)

    metrics_dict = {
        'hparam/best_f1': best_f1,
        'hparam/best_accuracy': val_metrics['accuracy'],
        'hparam/best_precision': val_metrics['precision'],
        'hparam/best_recall': val_metrics['recall']
    }

    writer.add_hparams(hparams, metrics_dict)
    writer.close()
    return model


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = f"{ENET_MODEL_TRAIN_LOG_DIR}/{timestamp}/model_batch_{ENET_BATCH_SIZE}_lr{ENET_LEARNING_RATE:.0e}"
    writer = SummaryWriter(log_dir)

    seed_everything(SEED)  # Assuming seed is constant or passed

    train_model(writer)

    print("TensorBoard logs saved!")
    print(f"Run: tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main()
    utils_cuda_clear()
