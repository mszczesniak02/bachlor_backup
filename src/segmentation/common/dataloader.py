import os
import cv2
import torch
import numpy as np
import albumentations as A
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from segmentation.common.hparams import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # Elastyczne deformacje - kluczowe dla nauki ciągłości rys
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)

# Dla walidacji tylko resize i normalizacja (bez losowości!)
val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)


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

        # 3. Binaryzacja (Robusta)
        # Obsługa masek [0, 255] oraz [0, 1]
        max_val = mask.max()
        if max_val > 1:
            mask = mask / 255.0

        mask = (mask > 0.5).astype(np.float32)

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
                           mask_path=MASK_TRAIN_PATH, transform=train_transform)

    # Zbiór walidacyjny BEZ augmentacji (tylko resize/normalize)
    valid_ds = dataset_get(img_path=IMG_TEST_PATH,
                           mask_path=MASK_TEST_PATH, transform=val_transform)

    train_dl = dataloader_get(train_ds, is_training=True, bsize=batch_size)
    valid_dl = dataloader_get(valid_ds, is_training=False, bsize=batch_size)

    return (train_dl, valid_dl)


def main():
    print("nothing to do.")


if __name__ == "__main__":
    main()
