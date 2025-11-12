import matplotlib.pyplot as plt
import numpy as np

import torch


def plot_effect(image, mask, effect=[], effect_title="Transform"):
    if len(effect) != 0:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image)
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
