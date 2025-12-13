import torch
import torch.nn as nn
import os
from hparams import *


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # Encoder
        # Input: [B, 3, 256, 256]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2,
                      padding=1),  # -> [32, 128, 128]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1),  # -> [64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1),  # -> [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2,
                      padding=1),  # -> [256, 16, 16]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               # -> [128, 32, 32]
                               padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # -> [64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               # -> [32, 128, 128]
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
                               # -> [3, 256, 256]
                               padding=1, output_padding=1),
            # No activation here because inputs are normalized (can be negative)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def model_init():
    model = CAE()
    return model


def model_load(filepath, device=DEVICE):
    model = model_init()
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
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
