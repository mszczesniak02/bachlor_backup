# autopep8: off
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import seaborn as sns

# -------------------------importing common and utils -----------------------------

original_sys_path = sys.path.copy()

# moving to "src/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

# importing commons
from autoencoder.dataloader import *
from autoencoder.model import *
from autoencoder.hparams import *

# importing utils
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path

# --------------------------------------------------------------------------------

def load_model(model_path):
    model = model_init()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_image_paths(root_dir):
    # Expecting structure: root_dir/Negative (Healthy) and root_dir/Positive (Cracked)
    # Or any structure where we can distinguish.
    # Assuming kaggle-set structure: Negative = Healthy (0), Positive = Cracked (1)

    negative_paths = glob(os.path.join(root_dir, "Negative", "*.jpg"))
    positive_paths = glob(os.path.join(root_dir, "Positive", "*.jpg"))

    # If folders are named differently (e.g. 0_brak, 1_wlosowe etc from classification task)
    if not negative_paths and not positive_paths:
        print("Standard Negative/Positive folders not found. Checking for other structures...")
        # Check for 0_brak (Healthy) vs others (Cracked)
        negative_paths = glob(os.path.join(root_dir, "0_brak", "**", "*.jpg"), recursive=True)

        # All other folders are positive
        all_files = glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
        positive_paths = list(set(all_files) - set(negative_paths))

    return negative_paths, positive_paths

def compute_scores(model, image_paths, label):
    scores = []
    labels = []

    transform = get_val_transforms()

    # Create a temporary dataset/dataloader for batch processing (faster)
    class TempDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            img = cv2.imread(path)
            if img is None:
                # Return dummy or handle error. For simplicity, skip or error.
                # To keep batching simple, we'll raise error or return zeros (and filter later?)
                # Let's assume images are valid.
                raise ValueError(f"Invalid image: {path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=img)
            return augmented['image']

    dataset = TempDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    criterion = nn.MSELoss(reduction='none') # We want per-image loss

    with torch.no_grad():
        for images in tqdm(dataloader, desc=f"Processing label {label}"):
            images = images.to(DEVICE)
            outputs = model(images)

            # Compute MSE per image: [B, C, H, W] -> [B]
            loss = torch.mean((images - outputs) ** 2, dim=[1, 2, 3])

            scores.extend(loss.cpu().numpy())
            labels.extend([label] * len(images))

    return scores, labels

def plot_histograms(neg_scores, pos_scores, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(neg_scores, color='green', label='Healthy (Negative)', kde=True, stat="density", alpha=0.5)
    sns.histplot(pos_scores, color='red', label='Cracked (Positive)', kde=True, stat="density", alpha=0.5)
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MSE Loss")
    plt.ylabel("Density")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(labels, scores, save_path=None):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
    plt.show()

    return roc_auc, fpr, tpr, thresholds

def find_optimal_threshold(fpr, tpr, thresholds):
    # Youden's J statistic
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print(f"Best Threshold: {best_thresh:.6f} (TPR={tpr[ix]:.3f}, FPR={fpr[ix]:.3f})")
    return best_thresh

def main():
    model_path = f"{MODEL_DIR}/autoencoder_best.pth"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = load_model(model_path)

    print(f"Scanning {TEST_DIR}...")
    neg_paths, pos_paths = get_image_paths(TEST_DIR)

    print(f"Found {len(neg_paths)} negative (healthy) and {len(pos_paths)} positive (cracked) images.")

    if not neg_paths and not pos_paths:
        print("No images found.")
        return

    # Compute scores
    # Label 0 for Healthy (Negative), 1 for Cracked (Positive)
    # Anomaly detection: Higher score -> Anomaly (1)

    neg_scores, neg_labels = compute_scores(model, neg_paths, 0)
    pos_scores, pos_labels = compute_scores(model, pos_paths, 1)

    all_scores = neg_scores + pos_scores
    all_labels = neg_labels + pos_labels

    # Visualization
    os.makedirs(LOG_DIR, exist_ok=True)

    print("Plotting histograms...")
    plot_histograms(neg_scores, pos_scores, save_path=f"{LOG_DIR}/histogram_mse.png")

    if len(pos_scores) > 0:
        print("Computing ROC-AUC...")
        roc_auc, fpr, tpr, thresholds = plot_roc_curve(all_labels, all_scores, save_path=f"{LOG_DIR}/roc_curve.png")
        print(f"ROC AUC: {roc_auc:.4f}")

        best_thresh = find_optimal_threshold(fpr, tpr, thresholds)

        # Confusion Matrix at best threshold
        preds = [1 if s > best_thresh else 0 for s in all_scores]
        cm = confusion_matrix(all_labels, preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Cracked'], yticklabels=['Healthy', 'Cracked'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix (Thresh={best_thresh:.4f})')
        plt.savefig(f"{LOG_DIR}/confusion_matrix.png")
        plt.show()

    else:
        print("Skipping ROC-AUC (no positive samples).")

if __name__ == "__main__":
    main()
