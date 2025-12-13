import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

from hparams import *


class ConcreteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filter for image files only

        all_files = glob(root_dir+"*.*")

        valid_extensions = {'.jpg', '.jpeg', '.png'}
        self.image_paths = [f for f in all_files if os.path.splitext(
            f)[1].lower() in valid_extensions]

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(
                f"Failed to load image at path: {img_path}. Check if file exists and is not corrupted.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Basic transform if none provided
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        # For Autoencoder, Input = Target
        return image, image


def get_transforms():
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def dataloader_init(batch_size=BATCH_SIZE, num_workers=WORKERS):
    train_dataset = ConcreteDataset(
        root_dir=TRAIN_DIR,
        transform=get_transforms()
    )

    val_dataset = ConcreteDataset(
        root_dir=TEST_DIR,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader
