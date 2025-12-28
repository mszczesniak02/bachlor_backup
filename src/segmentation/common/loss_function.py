import torch
import torch.nn
import torch.nn.functional as F
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


class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(DiceBCELoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets, smooth=1e-6):
        # Stabilne numerycznie BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        inputs_sig = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_sig * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / \
            (inputs_sig.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice_score

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


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


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # mniejsza kara za FP (szum)
        self.beta = beta   # większa kara za FN (przerwy w pęknięciach)
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha *
                                        FP + self.beta * FN + self.smooth)
        return 1 - tversky


def main():
    print("nothing to do.")


if __name__ == "__main__":
    main()
