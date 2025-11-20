from torch.cuda import empty_cache
import matplotlib.pyplot as plt
import numpy as np

import torch
import random
import os
import gc


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def utils_cuda_clear():
    """
    Clear the leftover memory from COLAB training model
    """

    print("Clearning memory...", end="")
    empty_cache()
    gc.collect()
    empty_cache()
    print("done.")


def plot_effect(image, mask, effect=[], effect_title="Transform"):
    """
    Plots 2 images if no effect is passed.
    """
    if len(effect) != 0:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Image")
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Mask")
        ax[2].imshow(effect, cmap="gray")
        ax[2].set_title(effect_title)
    else:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Mask")

    plt.show()


def np2ten(img: np.array) -> torch.tensor:
    """
    Numpy array to torch tensor
    """
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
    Torch tensor to numpy array

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
