
# autopep8: off
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


def evaluate_dataset(model, dataloader, device, class_names):
    """
    Evaluates on the entire dataset
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
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
    print("EVALUATION RESULTS")
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

    return accuracy, all_preds, all_labels, all_confidences


def main():
    class_names = ["no_crack", "crack"]
    device = torch.device(DEVICE)

    print("Loading model...")
    # Assuming best_model.pth is the target
    model_path = f"{MODEL_DIR}/best_model.pth"
    model = model_load(model_path, device)
    print(f"Model loaded from: {model_path}")
    print(f"Device: {device}")

    # Use Test set for inference
    test_loader = dataloader_get(
        TEST_DIR, batch_size=ENTRY_BATCH_SIZE, is_training=False, num_workers=WORKERS)

    print(f"\nEvaluating on Test set...")
    evaluate_dataset(model, test_loader, device, class_names)

    # Visualization on few random samples
    print("\nShowing random sample predictions...")
    # Raw dataset for path access
    dataset = EntryDataset(TEST_DIR, transform=None)

    if len(dataset) > 0:
        indices = np.random.choice(
            len(dataset), size=min(3, len(dataset)), replace=False)
        for idx in indices:
            img_path, label = dataset.samples[idx]
            pred_class, conf, probs = predict_single(
                model, img_path, device, class_names)

            print(f"\nSample: {img_path}")
            print(
                f"True: {class_names[label]}, Predicted: {class_names[pred_class]}, Confidence: {conf*100:.2f}%")
            # visualize_prediction(img_path, pred_class, conf, probs, class_names)
    else:
        print("No samples found in test directory.")


if __name__ == "__main__":
    main()
