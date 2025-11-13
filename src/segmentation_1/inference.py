from hparams import *
from model import *
from dataloader import *

import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np

from plots import plot_effect
from post_seg_cleanup import cleanup


def main():

    model = model_load()
    model.eval()

    # get the test dataset

    dataset = dataset_get(img_path="../../../datasets/multi/test_img/",
                          mask_path="../../../datasets/multi/test_lab/", transform=transform_val)
    dataloader = dataloader_get(dataset, is_training=False)
    # metrics, predictions, ground_truths = evaluate_model(
    #     model, dataloader, DEVICE)

    magic = 960

    i, msk = dataset[magic]
    img = ten2np(i)
    msk = ten2np(msk)
    out = ten2np(model(i.unsqueeze(0)))

    # plot_effect(img, msk, effect=out, effect_title="output")
    post = cleanup(out)

    # plt.imshow(post, cmap="jet")
    # plt.show()
    plot_effect(msk, out, effect=post, effect_title="Segment pipeline")


if __name__ == "__main__":
    main()
