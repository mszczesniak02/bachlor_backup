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


model_path_only_DeepCrack = r"/home/krzeslaav/Projects/bachlor/model_tests/ONLY_DEEPCRACK/segformermodel_segformer_0.647862974802653.pth"
model_path_full_ds = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/segformermodel_segformer_0.5864474233337809.pth"

def main():

    model = model_load("segformer", filepath=model_path_only_DeepCrack)
    model.eval()

    dataset = dataset_get(img_path="../../../../datasets/dataset_segmentation/test_img/",
                          mask_path="../../../../datasets/dataset_segmentation/test_lab/", transform=val_transform)

    magic = 2

    img, msk, out = model_predict(model, dataset, magic)

    plot_effect(img, msk, effect=out, effect_title="output")


if __name__ == "__main__":
    main()
