from hparams import *
from model import *
from dataloader import *

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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


def train_epoch(model, loader, criterion, optimizer, device, writer, epoch, step):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

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


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc='Validating', leave=False)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
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


def main():
    class_names = ["0_brak", "1_wlosowe", "2_male", "3_srednie", "4_duze"]
    device = DEVICE

    train_loader = dataloader_get(
        TRAIN_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, is_training=True, num_workers=WORKERS
    )

    val_loader = dataloader_get(
        TEST_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, is_training=False, num_workers=WORKERS
    )

    model = model_init()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=SCHEDULER_PATIENCE)

    timestamp = datetime.now().strftime('%d_%H%M')
    log_dir = f"{TRAIN_LOG_DIR}/model_batch_{BATCH_SIZE}_lr{LEARNING_RATE:.0e}_{timestamp}"
    writer = SummaryWriter(log_dir)

    hparams = {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'image_size': IMAGE_SIZE,
        'weight_decay': WEIGHT_DECAY,
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'scheduler_patience': SCHEDULER_PATIENCE,
        'device': str(device),
        'model': 'EfficientNet-B0',
        'optimizer': 'AdamW',
        'loss_function': 'CrossEntropyLoss'
    }

    best_f1 = 0.0
    best_model_path = None
    epochs_without_improvement = 0
    step = 0

    print("Starting training...")
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    epoch_loop = tqdm(range(EPOCHS), desc='Epochs', leave=False)
    for epoch in epoch_loop:
        train_loss, train_metrics, step, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch, step
        )

        val_loss, val_metrics, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']

        plot_confusion_matrix(train_labels, train_preds,
                              class_names, writer, epoch, 'train/confusion_matrix')
        plot_confusion_matrix(val_labels, val_preds, class_names,
                              writer, epoch, 'val/confusion_matrix')

        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

        writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)

        writer.add_scalars('F1-Score', {
            'train': train_metrics['f1_score'],
            'val': val_metrics['f1_score']
        }, epoch)

        writer.add_scalars('Precision', {
            'train': train_metrics['precision'],
            'val': val_metrics['precision']
        }, epoch)

        writer.add_scalars('Recall', {
            'train': train_metrics['recall'],
            'val': val_metrics['recall']
        }, epoch)

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
            best_model_path = f"../../models/classification/model_f1_{best_f1:.4f}_epoch{epoch}.pth"
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
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

    print("TensorBoard logs saved!")
    print(f"Run: tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main()
