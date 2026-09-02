"""Losses and segmentation metrics."""
import torch
import torch.nn.functional as F

def dice_score(logits, target, eps=1e-7):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)

def iou_score(logits, target, eps=1e-7):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    dice = (2 * inter + 1) / (
        prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1
    )
    return bce + (1 - dice.mean())
