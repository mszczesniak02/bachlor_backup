from skimage.morphology import skeletonize          #
from skimage import data, morphology, measure       #
import matplotlib.pyplot as plt                     #


import numpy as np                                  #
from scipy import ndimage                           #

import os
from PIL import Image

# -------------------------------------------------------------


mask_path = "mask.jpg"
mask_img = "img.jpg"

np_mask = np.asarray(Image.open(mask_path))

print(np_mask)


plt.imshow(Image.open(mask_img))
plt.show()
