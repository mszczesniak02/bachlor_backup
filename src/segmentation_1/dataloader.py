import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np

import albumentations as A

import os

from hparams import *

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


transform_val = A.Compose(
    [A.Resize(height=512, width=512)]
)

transform_train = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.RandomBrightnessContrast(
        brightness_limit=0.1, contrast_limit=0.1, p=0.3),
])


class DeepCrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([os.path.join(img_dir, file)
                             for file in os.listdir(img_dir)])
        self.masks = sorted([os.path.join(mask_dir, file)
                            for file in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as img:
            img = img.convert('RGB')
            img = img.resize((512, 512), Image.BILINEAR)  # WYMUSZENIE 512x512
            np_image = np.array(img)

        with Image.open(self.masks[index]) as mask:
            mask = mask.convert('L')
            mask = mask.resize((512, 512), Image.NEAREST)  # WYMUSZENIE 512x512
            np_mask = np.array(mask)

        # np_image = np.array(Image.open(self.images[index]))
        # np_mask = np.array(Image.open(self.masks[index]))

        # if len(np_mask.shape) == 3:
        #     np_mask = np_mask[:, :, 0]

        np_mask = (np_mask > 127).astype(np.uint8)

        if self.transform:  # if using transforms
            t = self.transform(image=np_image, mask=np_mask)
            np_image = t["image"]
            np_mask = t["mask"]

        # conversion from numpy array convention to tensor via permute,
        #     then normalizing to [0,1] range, same for mask, only using binary data
        tensor_image = torch.from_numpy(
            np_image).permute(2, 0, 1).float() / 255.0
        tensor_mask = torch.from_numpy(np_mask).unsqueeze(0).float()

        return tensor_image, tensor_mask


def dataset_get(img_path=IMG_TRAIN_PATH, mask_path=MASK_TRAIN_PATH, transform=None):

    dataset = DeepCrackDataset(img_path, mask_path, transform=transform)
    return dataset


def dataloader_get(dataset, is_training=True):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=is_training, num_workers=WORKERS, pin_memory=PIN_MEMORY)
    return dataloader


def dataset_split(dataset: DeepCrackDataset, test_factor: float, val_factor: float) -> list:
    """Split exising dataset given percentages as [0,1] floats, return list of  """
    return random_split(dataset, [test_factor, val_factor])


def np2ten(img: np.array) -> torch.tensor:
    if len(img.shape) == 2:
        # Jeśli grayscale, dodaj wymiar kanału
        img = np.expand_dims(img, axis=-1)

        # Konwersja: [H, W, C] -> [C, H, W] i normalizacja do [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Dodaj wymiar batcha [1, C, H, W] - potrzebne dla modelu
        tensor = tensor.unsqueeze(0)

        return tensor


def ten2np(img: torch.Tensor) -> np.array:
    """
    Konwertuje torch tensor na numpy array (obraz)
    skeleton
    Args:
        img: tensor w formacie [1, C, H, W] lub [C, H, W]

    Returns:
        numpy array w formacie [H, W, C] z wartościami [0, 255]
    """
    # Usuń batch dimension jeśli istnieje
    if len(img.shape) == 4:
        img = img.squeeze(0)  # [1, C, H, W] -> [C, H, W]

    # Przenieś na CPU i konwertuj do numpy
    img = img.detach().cpu()

    # Konwersja: [C, H, W] -> [H, W, C]
    img = img.permute(1, 2, 0).numpy()

    # Denormalizacja: [0, 1] -> [0, 255]
    img = (img * 255).astype(np.uint8)

    # Jeśli grayscale (1 kanał), usuń wymiar kanału
    if img.shape[2] == 1:
        img = img.squeeze(2)

    return img


def dataset_to_numpy(dataset: DeepCrackDataset):
    imgs, msks = [], []
    for (img, msk) in (iter(dataset)):
        imgs.append(ten2np(img))
        msks.append(ten2np(msk))
    return (imgs), (msks)


def main():
    print("nothing to do.")


if __name__ == "__main__":
    main()
