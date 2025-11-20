from model import *
from hparams import *
from dataloader import *
from loss_function import *

import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# tuning params
epochs = 3
batch_sizes = [4, 8, 16, 32, 60]  # , 8, 16, 32, 60]
lrs = [10**(np.random.uniform() * -4.0) for _ in range(5)]


def calculate_metrics(predictions, targets, threshold=0.5):

    # Binaryzacja
    preds = (predictions > threshold).float()
    targets = targets.float()

    # Flatten
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # True/False Positives/Negatives
    TP = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
    TN = ((preds_flat == 0) & (targets_flat == 0)).sum().float()
    FP = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
    FN = ((preds_flat == 0) & (targets_flat == 1)).sum().float()

    conf_table = [[TP, FP], [FN, TN]]

    # Metryki
    epsilon = 1e-7  # Unikaj dzielenia przez zero

    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    specificity = TN / (TN + FP + epsilon)

    # IoU (Intersection over Union) - NAJWAŻNIEJSZA dla segmentacji!
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = intersection / (union + epsilon)

    # Dice Coefficient
    dice = (2 * intersection) / (preds.sum() + targets.sum() + epsilon)

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item(),
        'specificity': specificity.item(),
        'iou': iou.item(),
        'dice': dice.item(),
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


def main():
    print("Tuning hyperParams\n Batch sizes: Learning rates:")
    for i in range(len(lrs)):
        print(f"\t{batch_sizes[i]},\t   {lrs[i]:.4e}")

    # Writer główny dla wszystkich eksperymentów
    parent_writer = SummaryWriter(MODEL_TRAIN_LOG_DIR)

    for batch_idx, bsize in enumerate(batch_sizes, 1):
        train_ds = dataset_get(
            IMG_TRAIN_PATH, MASK_TRAIN_PATH, transform=transform_train)
        val_ds = dataset_get(IMG_TEST_PATH, MASK_TEST_PATH,
                             transform=transform_val)

        train_dl = dataloader_get(train_ds, bsize=bsize)
        val_dl = dataloader_get(val_ds, bsize=bsize, is_training=False)

        for lr_idx, lr in enumerate(lrs, 1):
            combo_num = (batch_idx - 1) * len(lrs) + lr_idx
            combo_tot = len(batch_sizes) * len(lrs)

            writer = SummaryWriter(
                MODEL_TRAIN_LOG_DIR + f"/batch_{bsize}_lr_{lr:.4e}")

            model = model_init()
            model.to(DEVICE)
            model.train()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=WEIGHT_DECAY,
            )
            criterion = DiceLoss()
            best_val_iou = 0.0
            best_epoch = 0
            patience_counter = 0

            epoch_pbar = tqdm(range(epochs),
                              desc=f"[Combo {combo_num}/{combo_tot}] Batch={bsize}, LR={lr:.2e}",
                              leave=False,
                              position=0)

            for epoch in epoch_pbar:
                train_metrics = train_epoch(
                    model, train_dl, criterion, optimizer, DEVICE)
                val_metrics = validate(model, val_dl, criterion, DEVICE)

                current_lr = optimizer.param_groups[0]['lr']

                # Logowanie metryk do tensorboard
                writer.add_scalars('Loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss']
                }, epoch)

                writer.add_scalars('IoU', {
                    'train': train_metrics['iou'],
                    'val': val_metrics['iou']
                }, epoch)

                writer.add_scalars('Dice', {
                    'train': train_metrics['dice'],
                    'val': val_metrics['dice']
                }, epoch)

                writer.add_scalars('Precision', {
                    'train': train_metrics['precision'],
                    'val': val_metrics['precision']
                }, epoch)

                writer.add_scalars('Recall', {
                    'train': train_metrics['recall'],
                    'val': val_metrics['recall']
                }, epoch)

                writer.add_scalars('F1-Score', {
                    'train': train_metrics['f1_score'],
                    'val': val_metrics['f1_score']
                }, epoch)

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
                    model_save(model, checkpoint,
                               filename=f"model_{best_val_iou:.4f}.pth")
                else:
                    epoch_pbar.set_description(
                        f'[Combo {combo_num}/{combo_tot}] Best IoU: {best_val_iou:.4f}, No improvement: {patience_counter}/{EARLY_STOPPING_PATIENCE}')
                    patience_counter += 1

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n  Early stopping triggered!")
                    print(
                        f"   No improvement for {EARLY_STOPPING_PATIENCE} epochs")
                    print(
                        f"   Best IoU: {best_val_iou:.4f} at epoch {best_epoch + 1}")
                    break

            # Po zakończeniu treningu dla danej kombinacji hiperparametrów
            # Dodaj wyniki do tensorboard jako hparams
            hparam_dict = {
                'batch_size': bsize,
                'learning_rate': lr,
                'weight_decay': WEIGHT_DECAY,
            }

            metric_dict = {
                'hparam/best_val_iou': best_val_iou,
                'hparam/best_val_dice': val_metrics['dice'],
                'hparam/best_val_loss': val_metrics['loss'],
                'hparam/best_val_f1': val_metrics['f1_score'],
                'hparam/best_epoch': best_epoch,
            }

            writer.add_hparams(hparam_dict, metric_dict)
            writer.close()

    parent_writer.close()
    print("\nHyperparameter tuning completed!")


if __name__ == "__main__":
    main()
