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


def main():

    model = model_load("segformer", filepath="../../../../models/segformer/.pth")
    model.eval()

    dataset = dataset_get(img_path="../../../../datasets/multi/test_img/",
                          mask_path="../../../../datasets/multi/test_lab/", transform=transform_val)

    magic = 960

    img, msk, out = model_predict(model, dataset)

    plot_effect(img, msk, effect=out, effect_title="output")


if __name__ == "__main__":
    main()
