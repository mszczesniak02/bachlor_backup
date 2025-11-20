from model import *
from dataloader import *
from loss_function import *
from hparams import *
from plots import *


def main() -> int:

    model_1 = model_load(
        "../../models/segmentation_1/model_dice.pth", DEVICE, False)
    model_2 = model_load(
        "../../models/segmentation_1/model_bce.pth", DEVICE, False)

    dataset = dataset_get()
    # val_dl = dataloader_get(val_ds)

    magic = 10

    i, msk = dataset[magic]
    img = ten2np(i)
    msk = ten2np(msk)
    out1 = ten2np(model_1(i.unsqueeze(0)))
    out2 = ten2np(model_2(i.unsqueeze(0)))

    # plot_effect(msk, out1, out2)

    # plot_effect(msk, out, effect=post, effect_title="Segment pipeline")
    out_mean = (out1 + out2) / 2

    plot_effect(out1, out2, effect=out_mean)

    return 0


if __name__ == "__main__":
    main()
