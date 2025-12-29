#autopep8: off
import sys
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

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


def get_samples_per_class(dataset, num_samples_per_class=4):
    """
    Collects num_samples random samples for each class
    """
    samples_by_class = {i: [] for i in range(NUM_CLASSES)}
    class_indices = {i: [] for i in range(NUM_CLASSES)}

    # Scan dataset
    print("Scaling dataset for class samples...")
    for idx, (img_path, label) in enumerate(dataset.samples):
        if len(samples_by_class[label]) < num_samples_per_class * 5: # Collect pool
            samples_by_class[label].append(img_path)

    # Select random
    final_samples = {}
    for cls in range(NUM_CLASSES):
        if len(samples_by_class[cls]) >= num_samples_per_class:
            final_samples[cls] = np.random.choice(samples_by_class[cls], num_samples_per_class, replace=False)
        else:
            final_samples[cls] = samples_by_class[cls] # Take all if not enough

    return final_samples

def get_transform(device, image_size=IMAGE_SIZE):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

def predict_single(model, image_path, device, transform):
    model.eval()
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence

def visualize_grid_by_class(model, dataset, device, class_names):
    """
    Visualizes 4x4 grid: Rows are classes, columns are random samples.
    """
    transform = get_transform(device)

    # Get samples
    samples = get_samples_per_class(dataset, num_samples_per_class=4)

    # Create plot
    fig, axes = plt.subplots(NUM_CLASSES, 4, figsize=(20, 5 * NUM_CLASSES))
    # Handle case if NUM_CLASSES is not 4 or axes is not 2D
    if NUM_CLASSES == 1: axes = np.array([axes])
    if len(axes.shape) == 1: axes = axes.reshape(NUM_CLASSES, 4) if NUM_CLASSES > 1 else axes.reshape(1, 4)

    for cls_idx in range(NUM_CLASSES):
        cls_name = class_names[cls_idx]
        cls_samples = samples[cls_idx]

        for col_idx in range(4):
            ax = axes[cls_idx, col_idx]

            if col_idx < len(cls_samples):
                img_path = cls_samples[col_idx]

                # Predict
                pred_cls, conf = predict_single(model, img_path, device, transform)

                # Load image for display
                img_display = Image.open(img_path).convert('RGB')

                ax.imshow(img_display)

                is_correct = (pred_cls == cls_idx)
                color = 'green' if is_correct else 'red'

                title = f"Prawdziwa: {cls_name}\nPredykcja: {class_names[pred_cls]}\n({conf*100:.1f}%)"
                # ax.set_title(title, color=color, fontsize=10, fontweight='bold')

                # Draw text on image
                ax.text(10, 10, title, color=color, fontsize=8, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), va='top', ha='left')
            else:
                ax.text(0.5, 0.5, "Brak próbki", ha='center')

            ax.axis('off')

            # Label the row on the left
            if col_idx == 0:
                ax.text(-0.2, 0.5, f"Klasa {cls_idx}\n{cls_name}", rotation=90, 
                        va='center', ha='right', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

def main():
    np.random.seed(42)  # Ensure consistent samples across models
    class_names = ["Włosowate", "Małe", "Średnie", "Duże"] # Adjust based on 4 classes mapping
    if NUM_CLASSES != len(class_names):
        print(f"Warning: Hparams NUM_CLASSES ({NUM_CLASSES}) != len(class_names) ({len(class_names)})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = "efficienet"
    model_path = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/classification/newer/efficientnet/efficientnet_f1_0.9134_epoch15.pth"

    # Load dataset structure only
    test_dataset = dataset_get(TEST_DIR, image_size=IMAGE_SIZE, is_training=False)

    print(f"\nProcessing {model_name}...")
    print("Loading model...")

    try:
        model = model_load(model_name, filepath=model_path, device=device)
        model.eval()
        print(f"Model loaded from: {model_path}")

        print(f"Visualizing predictions grid for {model_name}...")
        visualize_grid_by_class(model, test_dataset, device, class_names)
    except Exception as e:
        print(f"Failed to load or process {model_name}: {e}")




if __name__ == "__main__":
    main()

