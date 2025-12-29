# autopep8: off
import sys
import os
original_sys_path = sys.path.copy()
# moving to "segmentation/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))
# importing commons
from segmentation.common.dataloader import *
from segmentation.common.model import *
from segmentation.common.hparams import *
# importing utils
from utils.utils import *
# go back to the origin path
sys.path = original_sys_path
# normal imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
# autopep8: off


model_path_full_ds = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/model_unet_0.5960555357910763.pth"
model_path_only_DeepCrack = "/home/krzeslaav/Projects/bachlor/model_tests/ONLY_DEEPCRACK/model_unet_0.7051329215367635.pth"

def main():

    model = model_load("unet", filepath=model_path_full_ds)
    model.eval()

    dataset = dataset_get(img_path="../../../../datasets/dataset_segmentation/test_img/",
                          mask_path="../../../../datasets/dataset_segmentation/test_lab/", transform=val_transform)

    magic = 1

    img, msk, out = model_predict(model, dataset, magic)

    plot_effect(img, msk, effect=out, effect_title="output")

if __name__ == "__main__":
    main()
