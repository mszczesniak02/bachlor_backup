# not to be run from here, only imported
from classification.common.hparams import *

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os
from pathlib import Path


class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        self.class_names = ["0_brak", "1_wlosowe",
                            "2_male", "3_srednie", "4_duze"]

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name / "masks"

            if not class_dir.exists():
                # Fallback to checking class directory implementation
                class_dir = self.root_dir / class_name
                if not class_dir.exists():
                    continue

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load as RGB (3 channels) to satisfy model requirements
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def get_class_distribution(self):
        distribution = {i: 0 for i in range(5)}
        for _, label in self.samples:
            distribution[label] += 1
        return distribution


def get_transforms(image_size=DEFAULT_IMAGE_SIZE, is_training=True):
    if is_training:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


def dataset_get(root_dir, image_size=DEFAULT_IMAGE_SIZE, is_training=True):
    transform = get_transforms(image_size, is_training)
    dataset = CrackDataset(root_dir, transform=transform)
    return dataset


def dataloader_get(root_dir, batch_size=DEFAULT_BATCH_SIZE, image_size=DEFAULT_IMAGE_SIZE, is_training=True, num_workers=WORKERS):
    dataset = dataset_get(root_dir, image_size, is_training)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    return dataloader


def dataloader_init(batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    # Using global TRAIN_DIR and TEST_DIR from hparams
    train_dl = dataloader_get(
        TRAIN_DIR, batch_size=batch_size, is_training=True)
    valid_dl = dataloader_get(
        TEST_DIR, batch_size=batch_size, is_training=False)

    return train_dl, valid_dl


def main():
    print("nothing to do")


if __name__ == "__main__":
    main()
