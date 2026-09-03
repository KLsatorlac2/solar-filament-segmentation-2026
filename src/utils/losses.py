"""Loss functions and segmentation metrics."""

import torch
import torch.nn.functional as F


def dice_score(logits, target, eps=1e-7):
    """Calculate binary Dice score using a 0.5 threshold."""
    pred = (torch.sigmoid(logits) > 0.5).float()

    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum()

    return (2 * intersection + eps) / (denominator + eps)


def iou_score(logits, target, eps=1e-7):
    """Calculate binary IoU score using a 0.5 threshold."""
    pred = (torch.sigmoid(logits) > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + eps) / (union + eps)


def dice_loss(logits, target, eps=1.0):
    """Calculate differentiable soft Dice loss."""
    prob = torch.sigmoid(logits)

    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (denominator + eps)

    return 1 - dice.mean()


def bce_dice_loss(logits, target):
    """Calculate the baseline BCE + Dice loss."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target)

    return bce + dice


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    """
    Calculate binary Focal Loss.

    Focal Loss reduces the contribution of easy pixels
    and focuses training on difficult pixels.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    prob = torch.sigmoid(logits)

    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    focal_weight = alpha_t * (1 - p_t).pow(gamma)

    return (focal_weight * bce).mean()


def focal_dice_loss(logits, target):
    """Calculate Focal Loss + Dice Loss."""
    focal = focal_loss(logits, target)
    dice = dice_loss(logits, target)

    return focal + dice


def tversky_loss(
    logits,
    target,
    alpha=0.3,
    beta=0.7,
    eps=1.0,
):
    """
    Calculate Tversky Loss.

    alpha controls the penalty for false positives.
    beta controls the penalty for false negatives.
    """
    prob = torch.sigmoid(logits)

    true_positive = (prob * target).sum(dim=(1, 2, 3))
    false_positive = (prob * (1 - target)).sum(dim=(1, 2, 3))
    false_negative = ((1 - prob) * target).sum(dim=(1, 2, 3))

    tversky = (
        true_positive + eps
    ) / (
        true_positive
        + alpha * false_positive
        + beta * false_negative
        + eps
    )

    return 1 - tversky.mean()


def get_loss_function(name):
    """Return a loss function by name."""
    loss_functions = {
        "bce_dice": bce_dice_loss,
        "focal": focal_loss,
        "focal_dice": focal_dice_loss,
        "tversky": tversky_loss,
    }

    if name not in loss_functions:
        available = ", ".join(loss_functions.keys())
        raise ValueError(
            f"Unknown loss: {name}. Available losses: {available}"
        )

    return loss_functions[name]