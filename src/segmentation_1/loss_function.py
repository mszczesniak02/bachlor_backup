import torch
import torch.nn
from torch.nn.functional import binary_cross_entropy
from torchvision.ops import sigmoid_focal_loss


class BCrossEntropyLoss(torch.nn.Module):
    def __init__(self, smooth=1e-16):  # zmiana
        super(BCrossEntropyLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        loss = binary_cross_entropy(predictions, targets)
        return loss


class FocalLoss(torch.nn.Module):

    def __init__(self, smooth=1e-16):  # zmiana
        super(BCrossEntropyLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        loss = sigmoid_focal_loss(predictions, targets)
        return loss


class BCE_with_FocalLoss(torch.nn.Module):
    def __init__(self, smooth=1e-16):  # zmiana
        super(BCE_with_FocalLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # loss_focal = sigmoid_focal_loss(predictions,targets)
        # loss_bce = binary_cross_entropy(predictions, targets)
        loss = 0.5 * sigmoid_focal_loss(predictions, targets) + \
            0.5 * binary_cross_entropy(predictions, targets)

        return loss


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / \
            (predictions.sum() + targets.sum() + self.smooth)

        return 1-dice


class DiceFocalLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        focal_loss = sigmoid_focal_loss(predictions, targets)

        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / \
            (predictions.sum() + targets.sum() + self.smooth)

        dice_loss = 1-dice

        the_loss = dice_loss * 0.5 + focal_loss * 0.5

        return the_loss


def main():
    print("nothing to do.")


if __name__ == "__main__":
    main()
