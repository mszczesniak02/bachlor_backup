# autopep8: off
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch
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


def validate(model, loader, criterion, device):
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


def tune_single_model(bsize, lr, epochs, global_pbar, device=DEVICE):
    class_names = ["0_brak", "1_wlosowe", "2_male", "3_srednie", "4_duze"]

    run_name = f"bs_{bsize}_lr_{lr:.1e}"
    writer = SummaryWriter(CONV_MODEL_TRAIN_LOG_DIR + "/tuning_" +
                           datetime.now().strftime('%H%M') + "/" + run_name)

    train_loader, val_loader = dataloader_init(batch_size=bsize)

    model = model_init(model_name="convnet")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=CONV_WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=CONV_SCHEDULER_PATIENCE)

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    step = 0

    for epoch in range(epochs):
        global_pbar.set_description(
            f"Run: {run_name} | Epoch {epoch+1}/{epochs} | Best F1: {best_f1:.4f}")

        train_loss, train_metrics, step, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch, step
        )

        val_loss, val_metrics, val_labels, val_preds = validate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        metrics_to_log = ['accuracy', 'f1_score', 'precision', 'recall']

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
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        global_pbar.update(1)

        if patience_counter >= PATIENCE:
            skipped_epochs = epochs - (epoch + 1)
            if skipped_epochs > 0:
                global_pbar.update(skipped_epochs)
            break

    hparam_dict = {
        'batch_size': bsize,
        'learning_rate': lr,
        'weight_decay': CONV_WEIGHT_DECAY,
    }

    metric_dict = {
        'hparam/best_f1': best_f1,
        'hparam/best_epoch': best_epoch,
    }

    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    return best_f1, best_epoch


def run_tuning(batch_sizes, learning_rates, epochs=3):
    print("Tuning hyperParams")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Learning rates: {learning_rates}")

    total_epochs = len(batch_sizes) * len(learning_rates) * epochs

    with tqdm(total=total_epochs, desc="Total Tuning Progress") as global_pbar:
        for bsize in batch_sizes:
            for lr in learning_rates:
                tune_single_model(bsize, lr, epochs, global_pbar)

    print("\nHyperparameter tuning completed!")


if __name__ == "__main__":
    batch_sizes = [16, 8, 4]
    learning_rates = [1.3e-3, 2.5e-4, 4.5e-5]
    run_tuning(batch_sizes, learning_rates)
    utils_cuda_clear()
