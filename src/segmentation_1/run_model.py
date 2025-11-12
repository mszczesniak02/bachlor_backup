import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import albumentations as A
import time

def visualize_prediction(image_path, prediction, threshold=0.5):
    """
    Wizualizacja predykcji
    
    Args:
        image_path: ścieżka do oryginalnego obrazu
        prediction: maska prawdopodobieństwa [H, W]
        threshold: próg binaryzacji (default: 0.5)
    """
    # Wczytaj oryginalny obraz
    image = np.array(Image.open(image_path))
    
    # Binaryzacja predykcji
    binary_mask = (prediction > threshold).astype(np.uint8)
    
    # Wizualizacja
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Oryginalny obraz
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Heatmapa prawdopodobieństwa
    im1 = axes[1].imshow(prediction, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Probability Heatmap", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Binarna maska
    axes[2].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Binary Mask (threshold={threshold})", fontsize=14)
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(image)
    axes[3].imshow(prediction, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[3].set_title("Overlay", fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Statystyki
    crack_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    crack_percentage = (crack_pixels / total_pixels) * 100
    
    print(f"📊 Statistics:")
    print(f"   Crack pixels: {crack_pixels:,} ({crack_percentage:.2f}% of image)")
    print(f"   Max probability: {prediction.max():.3f}")
    print(f"   Mean probability: {prediction.mean():.3f}")


def predict_single_image(image_path, model, device):
    """Predykcja na jednym obrazie"""
    # Wczytaj obraz
    image = np.array(Image.open(image_path))
    original_shape = image.shape[:2]
    
    # Resize do 256x256 (rozmiar treningowy)
    transform = A.Compose([A.Resize(height=256, width=256)])
    image_resized = transform(image=image)['image']
    
    # Konwersja do tensora
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 256, 256]
    
    # Predykcja
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
    
    # Konwersja do numpy
    prediction = prediction.squeeze().cpu().numpy()  # [256, 256]
    
    # Resize do oryginalnego rozmiaru
    resize_back = A.Resize(height=original_shape[0], width=original_shape[1])
    prediction = resize_back(image=prediction)['image']
    
    return prediction

# ========================================
# KROK 1: Utwórz TAKI SAM model (architektura)
# ========================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # ← NIE ładuj ImageNet, załadujesz swoje wagi
    in_channels=3,
    classes=1,
    activation=None,
)

# ========================================
# KROK 2: Wczytaj zapisane wagi
# ========================================
checkpoint = torch.load('../models/unet_epoch3.pth', map_location='cpu')


model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

print("✅ Model loaded!")
print(f"   Epoch: {checkpoint['epoch']}")
print(f"   Val Loss: {checkpoint['val_loss']:.4f}")


# ========================================
# KROK 3: Przenieś na CPU lub GPU
# ========================================


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"   Device: {device}")



image_path = "/home/krzeslaav/bachlor/project/assets/datasets/DeepCrack/test_img/11125-1.jpg"
i2 = "/home/krzeslaav/bachlor/project/assets/datasets/DeepCrack/test_img/IMG24-3.jpg"

i3 = '/home/krzeslaav/bachlor/project/assets/datasets/DeepCrack/test_img/IMG24-2.jpg'


t_start = time.time()
prediction = predict_single_image(image_path,model,device)
print(f"   Max probability: {prediction.max():.3f}")
print(f"   Mean probability: {prediction.mean():.3f}")
print("Time 1: "+ str(time.time() - t_start))
t_start = time.time()
prediction = predict_single_image(i2,model,device)
print(f"   Max probability: {prediction.max():.3f}")
print(f"   Mean probability: {prediction.mean():.3f}")
print("Time 2: "+ str(time.time() - t_start))
t_start = time.time()

prediction = predict_single_image(i3,model,device)
print(f"   Max probability: {prediction.max():.3f}")
print(f"   Mean probability: {prediction.mean():.3f}")
print("Time 3: "+ str(time.time() - t_start))

visualize_prediction(image_path, prediction, threshold=0.5)