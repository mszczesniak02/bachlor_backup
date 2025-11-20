import torch
import torch.nn as nn
import torchvision.models as models

from hparams import *


def model_init():

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )

    return model


def model_load(filepath=MODEL_PATH, device=DEVICE):
    model = model_init()
    checkpoint = torch.load(filepath, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def model_save(model, filepath, epoch=None, optimizer=None, loss=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if loss is not None:
        checkpoint['loss'] = loss

    torch.save(checkpoint, filepath)


def main():
    print("nothing to do")


if __name__ == "__main__":
    main()
