import torch
import torch.nn as nn
import torch.nn.functional as F

def IoU(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))
    return ((inter + eps) / (union + eps)).mean()

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = (target > 0.5).float()
        inter = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()

def per_class_iou_mc(pred, target, num_classes: int, ignore_index: int = None, eps: float = 1e-6):
    if pred.dim() == 4 and pred.size(1) > 1:
        pred = pred.argmax(dim=1)
    if target.dim() == 4 and target.size(1) > 1:
        target = target.argmax(dim=1)
    pred = pred.long().view(-1)
    target = target.long().view(-1)
    if ignore_index is not None:
        mask = target != ignore_index
        pred, target = pred[mask], target[mask]
    ious = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            ious.append(float('nan'))
            continue
        pc = (pred == c)
        tc = (target == c)
        inter = (pc & tc).sum().item()
        union = pc.sum().item() + tc.sum().item() - inter
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((inter + eps) / (union + eps))
    return ious

def mean_iou_mc(pred, target, num_classes: int, ignore_index: int = None, eps: float = 1e-6):
    ious = per_class_iou_mc(pred, target, num_classes, ignore_index, eps)
    vals = [x for x in ious if x == x]
    return sum(vals) / max(len(vals), 1)

def macro_f1_mc(pred, target, num_classes: int, ignore_index: int = None, eps: float = 1e-6):
    if pred.dim() == 4 and pred.size(1) > 1:
        pred = pred.argmax(dim=1)
    if target.dim() == 4 and target.size(1) > 1:
        target = target.argmax(dim=1)
    pred = pred.long().view(-1)
    target = target.long().view(-1)
    if ignore_index is not None:
        mask = target != ignore_index
        pred, target = pred[mask], target[mask]
    f1s = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            f1s.append(float('nan'))
            continue
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        denom = (2*tp + fp + fn)
        if denom == 0:
            f1s.append(float('nan'))
        else:
            f1s.append((2*tp) / denom)
    vals = [x for x in f1s if x == x]
    return sum(vals) / max(len(vals), 1)

def organ_top1_acc(logits, target):
    if logits.dim() == 2:
        pred = logits.argmax(dim=1)
    else:
        pred = logits
    return (pred.long() == target.long()).float().mean()
