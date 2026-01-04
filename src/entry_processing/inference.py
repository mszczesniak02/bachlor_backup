
# autopep8: off
import sys
import os
import time
import resource
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

original_sys_path = sys.path.copy()
# moving to "src/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

from entry_processing.hparams import *
from entry_processing.model import *
from entry_processing.dataloader import dataloader_get, get_transforms, EntryDataset

sys.path = original_sys_path
# autopep8: on


def predict_single(model, image_path, device, class_names):
    """
    Predicts class for a single image
    """
    model.eval()

    # Manually loading image and applying transform
    # (reusing get_transforms logic but manually since we don't have dataset object per se)
    image = np.array(Image.open(image_path).convert('RGB'))

    transform = get_transforms(image_size=ENTRY_IMAGE_SIZE, is_training=False)
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()

    return predicted_class_idx, confidence, probabilities[0].cpu().numpy()


def visualize_prediction(image_path, predicted_class_idx, confidence, probabilities, class_names):
    """
    Visualizes image with prediction
    """
    image = Image.open(image_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(image)
    ax1.set_title(f'Predicted: {class_names[predicted_class_idx]}\nConfidence: {confidence*100:.2f}%',
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    colors = ['green' if i ==
              predicted_class_idx else 'gray' for i in range(len(class_names))]
    bars = ax2.barh(class_names, probabilities * 100, color=colors)
    ax2.set_xlabel('Probability (%)', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)

    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                 f'{prob*100:.1f}%', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()


def evaluate_dataset(model, dataloader, device, class_names, save_cm=False, subset_name="val"):
    """
    Evaluates on the entire dataset and computes confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f'Evaluating {subset_name} on {device}'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)
            confidences = probabilities.max(dim=1).values

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    avg_confidence = np.mean(all_confidences)

    print("\n" + "="*70)
    print(f"EVALUATION RESULTS ({subset_name.upper()} - {device})")
    print("="*70)
    print(f"Total samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Average confidence: {avg_confidence*100:.2f}%")
    print("="*70)

    for class_idx, class_name in enumerate(class_names):
        mask = np.array(all_labels) == class_idx
        if mask.sum() > 0:
            class_acc = (np.array(all_preds)[mask] == class_idx).mean()
            print(f"{class_name:15} : {class_acc*100:.2f}% ({mask.sum()} samples)")

    print("="*70)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix ({device}):")
    print(cm)

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'Confusion Matrix ({subset_name.upper()} - {device})')

    if save_cm:
        save_path = f"{subset_name}_confusion_matrix_{device}.png"
        plt.savefig(save_path)
        print(f"\nConfusion matrix plot saved to {save_path}")

    plt.close(fig)  # Close plot to free memory

    return accuracy, all_preds, all_labels, all_confidences


def measure_performance(device_name, model_path, val_loader, class_names):
    print(f"\n{'#'*30}")
    print(f"Testing on {device_name.upper()}")
    print(f"{'#'*30}")

    device = torch.device(device_name)

    # Load model
    print(f"Loading model to {device_name}...")
    model = model_load(model_path, device)

    # Reset memory stats
    if device_name == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.time()
    # Baseline memory
    if device_name == 'cuda':
        mem_start = torch.cuda.memory_allocated()
    else:
        mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Run evaluation
    evaluate_dataset(model, val_loader, device, class_names,
                     save_cm=True, subset_name="val")

    end_time = time.time()

    # Calculate metrics
    elapsed_time = end_time - start_time

    if device_name == 'cuda':
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = mem_peak  # Peak memory usage during the process
        mem_unit = "MB"
        mem_val = mem_used / 1024 / 1024
    else:
        # ru_maxrss is in kilobytes on Linux
        mem_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_used = mem_peak
        mem_unit = "MB"
        mem_val = mem_used / 1024  # Convert KB to MB

    print(f"\n{device_name.upper()} Performance:")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print(f"Peak Memory: {mem_val:.2f} {mem_unit}")

    return elapsed_time, mem_val, mem_unit


def main():
    class_names = ["no_crack", "crack"]
    model_path = f"{MODEL_DIR}/best_model.pth"

    # Evaluate on Validation set
    print(f"\nLoading Validation set...")
    # Using larger batch size implies more memory usage, keep consistent for benchmark
    val_loader = dataloader_get(
        VAL_DIR, batch_size=ENTRY_BATCH_SIZE, is_training=False, num_workers=WORKERS)

    results = {}

    # Test CUDA if available
    if torch.cuda.is_available():
        time_cuda, mem_cuda, unit_cuda = measure_performance(
            'cuda', model_path, val_loader, class_names)
        results['cuda'] = (time_cuda, mem_cuda, unit_cuda)
    else:
        print("\nCUDA not available, skipping CUDA test.")

    # Test CPU
    time_cpu, mem_cpu, unit_cpu = measure_performance(
        'cpu', model_path, val_loader, class_names)
    results['cpu'] = (time_cpu, mem_cpu, unit_cpu)

    # Summary
    print(f"\n{'='*20} BENCHMARK SUMMARY {'='*20}")
    print(f"{'Device':<10} | {'Time (s)':<15} | {'Memory':<15}")
    print("-" * 46)

    for device, (t, m, u) in results.items():
        print(f"{device.upper():<10} | {t:<15.4f} | {m:.2f} {u}")
    print("=" * 46)


if __name__ == "__main__":
    main()
