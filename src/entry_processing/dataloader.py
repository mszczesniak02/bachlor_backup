# not to be run from here, only imported
from pathlib import Path
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torch
from entry_processing.hparams import *
import sys
import os

original_sys_path = sys.path.copy()
# moving to "src/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))


sys.path = original_sys_path


class EntryDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        self.class_names = ["no_crack", "crack"]
        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist.")
                continue

            class_idx = self.class_to_idx[class_name]

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    self.samples.append((str(img_file), class_idx))

        if len(self.samples) == 0:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load as RGB
            image = np.array(Image.open(img_path).convert('RGB'))

            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']

            return image, label

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, ENTRY_IMAGE_SIZE, ENTRY_IMAGE_SIZE)), label


def get_transforms(image_size=ENTRY_IMAGE_SIZE, is_training=True):
    if is_training:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1,
                          saturation=0.1, hue=0.05, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


def dataloader_get(root_dir, batch_size=ENTRY_BATCH_SIZE, image_size=ENTRY_IMAGE_SIZE, is_training=True, num_workers=WORKERS):
    dataset = EntryDataset(
        root_dir, transform=get_transforms(image_size, is_training))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def dataloader_init(batch_size: int = ENTRY_BATCH_SIZE) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dl = dataloader_get(
        TRAIN_DIR, batch_size=batch_size, is_training=True)

    val_dl = dataloader_get(
        VAL_DIR, batch_size=batch_size, is_training=False)

    test_dl = dataloader_get(
        TEST_DIR, batch_size=batch_size, is_training=False)

    return train_dl, val_dl, test_dl
