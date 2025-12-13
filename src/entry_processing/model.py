
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from entry_processing.hparams import *


class SimpleClassificator(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleClassificator, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def model_init(num_classes=2):
    model = SimpleClassificator(num_classes=num_classes)
    return model


def model_load(filepath=None, device=DEVICE):
    model = model_init(num_classes=NUM_CLASSES)

    if filepath is None:
        filepath = f"{MODEL_DIR}/best_model.pth"

    if os.path.isfile(filepath):
        checkpoint = torch.load(
            filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(
            f"Warning: Model file {filepath} not found. Returning initialized model.")

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
