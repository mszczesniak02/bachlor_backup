import torch
import torch.nn as nn
import torchvision.models as models

from classification.common.hparams import *


def model_init(model_name: str):
    if model_name == "efficienet":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)

        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, NUM_CLASSES)
        )

        return model

    elif model_name == "convnet":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)

        # ConvNeXt classifier structure:
        # (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
        # (1): Flatten(start_dim=1, end_dim=-1)
        # (2): Linear(in_features=768, out_features=1000, bias=True)

        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, NUM_CLASSES)

        return model


def model_load(model_name: str, filepath=MODEL_PATH, device=DEVICE):
    model = model_init(model_name)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

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
