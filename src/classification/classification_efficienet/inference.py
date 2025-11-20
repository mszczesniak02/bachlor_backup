from hparams import *
from model import *
from dataloader import *

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image


def predict_single(model, image_path, device, class_names):
    """
    Przewiduje klasę dla pojedynczego obrazu
    """
    model.eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence, probabilities[0].cpu().numpy()


def visualize_prediction(image_path, predicted_class, confidence, probabilities, class_names):
    """
    Wizualizuje obraz z przewidywaniem
    """
    image = Image.open(image_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Predicted: {class_names[predicted_class]}\nConfidence: {confidence*100:.2f}%',
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    colors = ['green' if i ==
              predicted_class else 'gray' for i in range(len(class_names))]
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
    Ewaluacja na całym zbiorze danych
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

    per_class_acc = {}
    for class_idx, class_name in enumerate(class_names):
        mask = np.array(all_labels) == class_idx
        if mask.sum() > 0:
            class_acc = (np.array(all_preds)[mask] == class_idx).mean()
            per_class_acc[class_name] = class_acc
            print(f"{class_name:15} : {class_acc*100:.2f}% ({mask.sum()} samples)")

    print("="*70)

    return accuracy, all_preds, all_labels, all_confidences


def main():
    class_names = ["0_brak", "1_wlosowe", "2_male", "3_srednie", "4_duze"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    model = model_load(MODEL_PATH, device)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Device: {device}")

    test_dataset = dataset_get(
        TEST_DIR, image_size=IMAGE_SIZE, is_training=False)
    test_loader = dataloader_get(TEST_DIR, batch_size=32, image_size=IMAGE_SIZE,
                                 is_training=False, num_workers=WORKERS)

    print(f"\nTest dataset size: {len(test_dataset)}")

    accuracy, preds, labels, confidences = evaluate_dataset(
        model, test_loader, device, class_names
    )

    print("\nShowing random sample predictions...")
    random_indices = np.random.choice(len(test_dataset), size=3, replace=False)

    for idx in random_indices:
        sample_path = test_dataset.samples[idx][0]
        pred_class, confidence, probs = predict_single(
            model, sample_path, device, class_names
        )
        true_label = test_dataset.samples[idx][1]

        print(f"\nSample: {sample_path}")
        print(
            f"True: {class_names[true_label]}, Predicted: {class_names[pred_class]}, Confidence: {confidence*100:.2f}%")

        visualize_prediction(sample_path, pred_class,
                             confidence, probs, class_names)


if __name__ == "__main__":
    main()
