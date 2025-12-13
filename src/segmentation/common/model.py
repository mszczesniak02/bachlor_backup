# never run on its own, always imported
from segmentation.common.hparams import *
from utils.utils import *
#
import numpy as np
import torch
import segmentation_models_pytorch as smp               # preset model
import os


def model_init(model_name: str):
    if model_name == "unet":

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return model
    elif model_name == "segformer":
        model = smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return model


def model_load(model_name: str, filepath=MODEL_INFERENCE_PATH, device=DEVICE, print_info=False):
    model = model_init(model_name)

    if os.path.isfile(filepath):
        checkpoint = torch.load(
            filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        if print_info:
            print(f"Model loaded on {device}")
        return model
    else:
        print(f"ERROR: file {filepath} not found, returning empty model")
        return model


def model_save(model, checkpoint=None, filepath=MODEL_INFERENCE_DIR, filename="model.pth", print_info=False):
    if checkpoint == None:

        checkpoint = {'epoch': EPOCHS,
                      'model_state_dict': model.state_dict(),
                      }
    torch.save(checkpoint, filepath + filename)
    if print_info:
        print(f"Model saved to {filepath} as {filename}.")


def model_predict(model, dataset, index=69) -> np.array:
    i, msk = dataset[index]
    img = ten2np(i)
    msk = ten2np(msk)
    out = ten2np(model(i.unsqueeze(0)))
    return img, msk, out


def main():
    print("nothing to do.", end="")


if __name__ == "__main__":
    main()
