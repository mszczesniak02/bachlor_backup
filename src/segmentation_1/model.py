from hparams import *
from model import *

import torch

import segmentation_models_pytorch as smp               # preset model

import os


def model_init():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model


def model_load(filepath=MODEL_INFERENCE_PATH, device=DEVICE, print_info=False):
    model = model_init()

    if os.path.isfile(filepath):
        checkpoint = torch.load(
            filepath, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        if print_info:
            print(f"Model loaded on {device}")
        return model
    else:
        print("ERROR: file not found, returning empty model")
        return model


def model_save(model, checkpoint=None, filepath=MODEL_INFERENCE_DIR, filename="model.pth", print_info=False):
    if checkpoint == None:

        checkpoint = {'epoch': EPOCHS,
                      'model_state_dict': model.state_dict(),
                      }
    torch.save(checkpoint, filepath + filename)
    if print_info:
        print(f"Model saved to {filepath} as {filename}.")


def main():
    print("nothing to do.", end="")


if __name__ == "__main__":
    main()
