from hparams import *
from model import *
from dataloader import *

import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np

from utils import *


def main():

    model = model_load()
    model.eval()

    dataset = dataset_get(img_path="../../../../datasets/multi/test_img/",
                          mask_path="../../../../datasets/multi/test_lab/", transform=transform_val)

    magic = 960

    img, msk, out = model_predict(model, dataset)

    plot_effect(img, msk, effect=out, effect_title="output")


if __name__ == "__main__":
    main()
