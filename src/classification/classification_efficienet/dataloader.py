from hparams import *

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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
                continue

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), class_idx))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        distribution = {i: 0 for i in range(5)}
        for _, label in self.samples:
            distribution[label] += 1
        return distribution


def get_transforms(image_size=IMAGE_SIZE, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])


def get_dataloader(root_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, is_training=True, num_workers=WORKERS):
    transform = get_transforms(image_size, is_training)
    dataset = CrackDataset(root_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader, dataset


def dataset_get(root_dir, image_size=IMAGE_SIZE, is_training=True):
    transform = get_transforms(image_size, is_training)
    dataset = CrackDataset(root_dir, transform=transform)
    return dataset


def dataloader_get(root_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, is_training=True, num_workers=WORKERS):
    dataset = dataset_get(root_dir, image_size, is_training)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def main():
    print("nothing to do")


if __name__ == "__main__":
    main()
