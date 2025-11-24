import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from PIL import Image, ImageFile
import numpy as np
import albumentations as A

import os
# this is never run on as binary, so the line will not work on it's own
from segmentation.common.hparams import *


ImageFile.LOAD_TRUNCATED_IMAGES = True


transform_val = A.Compose(
    [A.Resize(height=512, width=512)]
)

# augmentation
transform_train = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.RandomBrightnessContrast(
        brightness_limit=0.1, contrast_limit=0.1, p=0.3),
])


class CrackDataset(Dataset):
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

        # When getting images from dirs, images differ in sizes and resolution.
        # To combat that issue, rescaling to 512x512px is being done for both masks and images.

        with Image.open(self.images[index]) as img:
            img = img.convert('RGB')
            # scale all images to 512x512 format
            img = img.resize((512, 512), Image.BILINEAR)
            np_image = np.array(img)

        with Image.open(self.masks[index]) as mask:
            mask = mask.convert('L')
            mask = mask.resize((512, 512), Image.NEAREST)
            np_mask = np.array(mask)

        # forcing binary map if map's vals are not 1' and 0's, but 255's and 0's.
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
    """
    Return train dataset (if no args passed)
    """

    dataset = CrackDataset(img_path, mask_path, transform=transform)
    return dataset


def dataset_split(dataset: CrackDataset, test_factor: float, val_factor: float) -> list:
    """Split exising dataset given percentages as [0,1] floats, return list of  """
    return random_split(dataset, [test_factor, val_factor])


def dataloader_get(dataset, is_training=True, bsize=DEFAULT_BATCH_SIZE):
    """
    Get dataloader from dataset.
    """
    dataloader = DataLoader(dataset, batch_size=bsize,
                            shuffle=is_training, num_workers=WORKERS, pin_memory=PIN_MEMORY)
    return dataloader


def dataloader_init(batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    """
        Get dataloader bypassing creating datasets and splitting, main purpose being less code in training function.
    """
    train_ds = dataset_get(img_path=IMG_TRAIN_PATH,
                           mask_path=MASK_TRAIN_PATH, transform=transform_train)

    valid_ds = dataset_get(img_path=IMG_TEST_PATH,
                           mask_path=MASK_TEST_PATH, transform=transform_val)

    train_dl = dataloader_get(train_ds, is_training=True, bsize=batch_size)
    valid_dl = dataloader_get(valid_ds, is_training=False, bsize=batch_size)

    return (train_dl, valid_dl)


def main():
    print("nothing to do.")


if __name__ == "__main__":
    main()
