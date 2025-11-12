from skimage.morphology import remove_small_objects, remove_small_holes, closing, skeletonize          #
from skimage import data, measure       #

import matplotlib.pyplot as plt                     #
import numpy as np                                  #

import os
from PIL import Image
import plots

# -------------------------------------------------------------

from plots import plot_effect, ten2np

# parametry wejściowe do dostrojenia, trzeba obliczyć średnie wartości które najlepiej odfiltrowywują zakłócenia


def cleanup(output, hole_area=100, min_object=50, connectivity=10):
    np_out = output

    for i, row in enumerate(np_out):
        for j, col in enumerate(row):
            if col < 150:
                np_out[i][j] = 0
            else:
                np_out[i][j] = 1

    mask_closed = closing(np_out, footprint=np.ones((3, 3)))

    mask_removed_holes = remove_small_holes(
        mask_closed, area_threshold=hole_area, connectivity=connectivity)
    mask_removed_objects = remove_small_objects(
        mask_removed_holes, min_size=min_object, connectivity=connectivity)

    return mask_removed_objects


def main():
    print("Nothing to do")


if __name__ == "__main__":
    main()
