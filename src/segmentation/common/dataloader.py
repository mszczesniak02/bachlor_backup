import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from PIL import Image, ImageFile
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
# this is never run on as binary, so the line will not work on it's own
from segmentation.common.hparams import *
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2  # <--- TEJ LINII BRAKUJE
import os
import cv2
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_transform = A.Compose([
    # --- Transformacje geometryczne (zmieniają kształt/położenie) ---
    # Upewnij się, że rozmiar pasuje do Twojego modelu
    A.Resize(height=256, width=256),
    A.HorizontalFlip(p=0.5),         # Lustrzane odbicie poziome
    A.VerticalFlip(p=0.5),           # Lustrzane odbicie pionowe
    A.Rotate(limit=35, p=0.5),       # Obrót o max 35 stopni

    # Skalowanie i przycinanie (uczy model widzieć obiekty w różnych rozmiarach)
    A.RandomScale(scale_limit=0.2, p=0.5),

    # Zniekształcenia (dobre, jeśli obiekty są organiczne/nieregularne)
    A.OneOf([
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        A.GridDistortion(p=1.0),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1.0),
    ], p=0.3),

    # --- Transformacje kolorystyczne i szum (tylko na obraz, nie maskę) ---
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20,
                             sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ], p=0.5),

    # Szum i rozmycie (symulacja gorszej jakości zdjęć)
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(p=1.0),
    ], p=0.3),

    # --- Normalizacja i konwersja do Tensora ---
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Dla walidacji tylko resize i normalizacja (bez losowości!)
val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Filtrujemy tylko pliki obrazów i sortujemy, żeby pary się zgadzały
        self.images = sorted([os.path.join(img_dir, f) for f in os.listdir(
            img_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(
            mask_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 1. Wczytanie obrazu (OpenCV)
        # CV2 wczytuje jako BGR, konwertujemy na RGB
        img_path = self.images[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Wczytanie maski (OpenCV)
        # Wczytujemy w skali szarości (0-255)
        mask_path = self.masks[index]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 3. Binaryzacja (Twoja logika - bardzo dobra dla DeepCrack)
        # DeepCrack ma wartości 0 i 255. Zamieniamy na 0 i 1.
        mask = (mask > 127).astype(np.uint8)

        # 4. Augmentacje (Albumentations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 'image' jest już Tensorem dzięki ToTensorV2 i jest znormalizowany (A.Normalize)

        # 'mask' wychodzi z Albumentations jako Tensor [H, W].
        # PyTorch oczekuje [Channels, H, W], więc dodajemy wymiar (unsqueeze).
        mask = mask.float().unsqueeze(0)

        return image, mask


def dataset_get(img_path=IMG_TRAIN_PATH, mask_path=MASK_TRAIN_PATH, transform=None):
    """
    Return train dataset
    """
    dataset = CrackDataset(img_path, mask_path, transform=transform)
    return dataset


def dataloader_get(dataset, is_training=True, bsize=DEFAULT_BATCH_SIZE):
    """
    Get dataloader from dataset.
    """
    dataloader = DataLoader(dataset, batch_size=bsize,
                            shuffle=is_training, num_workers=WORKERS, pin_memory=PIN_MEMORY)
    return dataloader


def dataloader_init(batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    """
    Get dataloader setup for DeepCrack (train/test split based on folders).
    """
    # Zbiór treningowy z augmentacją
    train_ds = dataset_get(img_path=IMG_TRAIN_PATH,
                           mask_path=MASK_TRAIN_PATH, transform=transform_train)

    # Zbiór walidacyjny BEZ augmentacji (tylko resize/normalize)
    valid_ds = dataset_get(img_path=IMG_TEST_PATH,
                           mask_path=MASK_TEST_PATH, transform=transform_val)

    train_dl = dataloader_get(train_ds, is_training=True, bsize=batch_size)
    valid_dl = dataloader_get(valid_ds, is_training=False, bsize=batch_size)

    return (train_dl, valid_dl)


def main():
    print("nothing to do.")


if __name__ == "__main__":
    main()
