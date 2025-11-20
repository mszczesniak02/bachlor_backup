from hparams import *
from model import *
from dataloader import *

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt


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

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }


def train_epoch(model, loader, criterion, optimizer, device, writer, epoch, global_step):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc='  Training', leave=False, position=1)
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

        if batch_idx % 10 == 0:
            global_step += 1
            writer.add_scalar('batch/train_loss', loss.item(), global_step)
            writer.add_scalar('batch/train_acc', acc, global_step)

    epoch_loss = running_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds, num_classes=NUM_CLASSES)

    return epoch_loss, metrics, global_step, all_labels, all_preds


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc='  Validating', leave=False, position=1)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_sizes = [16, 8, 4]
    learning_rates = [1.3e-3, 2.5e-4, 4.5e-5]

    EPOCHS_TUNING = 4
    IMAGE_SIZE_TUNING = 224

    timestamp = datetime.now().strftime('%m%d_%H%M')
    base_log_dir = f"{TRAIN_LOG_DIR}/hparams_tuning_{timestamp}"

    global_step = 0

    for bsize_idx, bsize in enumerate(batch_sizes, 1):
        for lr_idx, lr in enumerate(learning_rates, 1):
            combo_num = (bsize_idx - 1) * len(learning_rates) + lr_idx
            total_combos = len(batch_sizes) * len(learning_rates)

            log_dir = f"{base_log_dir}/batch_{bsize}_lr_{lr:.0e}"
            writer = SummaryWriter(log_dir)

            train_loader = dataloader_get(
                TRAIN_DIR, batch_size=bsize, image_size=IMAGE_SIZE_TUNING,
                is_training=True, num_workers=WORKERS
            )

            val_loader = dataloader_get(
                TEST_DIR, batch_size=bsize, image_size=IMAGE_SIZE_TUNING,
                is_training=False, num_workers=WORKERS
            )

            model = model_init()
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-4)

            run_train_losses = []
            run_train_accs = []
            run_train_f1s = []
            run_val_losses = []
            run_val_accs = []
            run_val_f1s = []

            epoch_pbar = tqdm(range(EPOCHS_TUNING),
                              desc=f"[Combo {combo_num}/{total_combos}] Batch={bsize}, LR={lr:.0e}",
                              leave=True,
                              position=0)

            for epoch in epoch_pbar:
                train_loss, train_metrics, global_step, train_labels, train_preds = train_epoch(
                    model, train_loader, criterion, optimizer, device, writer, epoch, global_step
                )

                val_loss, val_metrics, val_labels, val_preds = validate_epoch(
                    model, val_loader, criterion, device
                )

                run_train_losses.append(train_loss)
                run_train_accs.append(train_metrics['accuracy'])
                run_train_f1s.append(train_metrics['f1_score'])
                run_val_losses.append(val_loss)
                run_val_accs.append(val_metrics['accuracy'])
                run_val_f1s.append(val_metrics['f1_score'])

                epoch_pbar.set_postfix({
                    'epoch': f'{epoch+1}/{EPOCHS_TUNING}',
                    'val_acc': f'{val_metrics["accuracy"]*100:.2f}%',
                    'val_f1': f'{val_metrics["f1_score"]:.4f}',
                })

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

                writer.add_scalar('hyperparams/learning_rate', lr, epoch)
                writer.add_scalar('hyperparams/batch_size', bsize, epoch)

                plot_confusion_matrix(
                    val_labels, val_preds, class_names, writer, epoch, 'confusion_matrix'
                )

            writer.add_hparams(
                {
                    'batch_size': bsize,
                    'learning_rate': lr,
                    'image_size': IMAGE_SIZE_TUNING,
                    'epochs': EPOCHS_TUNING,
                    'model': 'EfficientNet-B0',
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-4
                },
                {
                    'hparam/train/final_loss': run_train_losses[-1],
                    'hparam/train/final_acc': run_train_accs[-1],
                    'hparam/train/final_f1': run_train_f1s[-1],
                    'hparam/val/final_loss': run_val_losses[-1],
                    'hparam/val/final_acc': run_val_accs[-1],
                    'hparam/val/final_f1': run_val_f1s[-1],
                    'hparam/val/best_acc': max(run_val_accs),
                    'hparam/val/best_f1': max(run_val_f1s),
                    'hparam/val/best_loss': min(run_val_losses),
                }
            )

            writer.close()

    print("\n" + "="*70)
    print("Hyperparameter tuning completed!")
    print(f"TensorBoard logs: {base_log_dir}")
    print(f"Run: tensorboard --logdir={base_log_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
