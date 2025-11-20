from skimage.morphology import remove_small_objects, remove_small_holes, closing, skeletonize          #
from skimage import data, measure       #

import matplotlib.pyplot as plt                     #
import numpy as np                                  #

import os
from PIL import Image

# -------------------------------------------------------------

from utils import plot_effect, ten2np


def cleanup(output, threshhold=0.1, min_hole=0.1, connectivity=10):
    np_out = ten2np(output)

    mask_closed = closing(np_out, footprint=np.ones((3, 3)))
    mask_removed_holes = remove_small_holes(
        mask_closed, area_threshold=threshhold, connectivity=connectivity)
    mask_removed_objects = remove_small_objects(
        mask_removed_holes, min_size=min_hole, connectivity=connectivity)

    return mask_removed_objects


def main():
    print("Nothing to do")


if __name__ == "__main__":
    main()
