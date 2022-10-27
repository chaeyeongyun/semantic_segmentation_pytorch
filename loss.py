import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import List

###### dice loss
def dice_coefficient(pred:torch.Tensor, target:torch.Tensor, num_classes:int, ignore_idx=None):
    assert pred.shape[0] == target.shape[0]
    epsilon = 1e-6
    if num_classes == 2:
        dice = 0
        # if both a and b are 1-D arrays, it is inner product of vectors(without complex conjugation)
        for batch in range(pred.shape[0]):
            pred_1d = pred[batch].view(-1)
            target_1d = target[batch].view(-1)
            inter = (pred_1d * target_1d).sum()
            sum_sets = pred_1d.sum() + target_1d.sum()
            dice += (2*inter+epsilon) / (sum_sets + epsilon)
        return dice / pred.shape[0]
        
    
    elif num_classes == 1:
        dice = 0
        pred = F.Sigmoid(pred)
        for batch in range(pred.shape[0]):
            pred_1d = pred[batch].view(-1)
            target_1d = target[batch].view(-1)
            inter = (pred_1d * target_1d).sum()
            sum_sets = pred_1d.sum() + target_1d.sum()
            dice += (2*inter+epsilon) / (sum_sets + epsilon)
        return dice / pred.shape[0]
        
    else:
        pred = F.softmax(pred, dim=1).float()
        dice = 0
        for c in range(num_classes):
            if c==ignore_idx:
                continue
            dice += dice_coefficient(pred[:, c, :, :], torch.where(target==c, 1, 0), 2, ignore_idx)
        return dice / num_classes 

def dice_loss(pred, target, num_classes, ignore_idx=None):
    if not isinstance(pred, torch.Tensor) :
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(pred)}")

    dice = dice_coefficient(pred, target, num_classes, ignore_idx)
    return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_idx=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
    
    def forward(self, pred, target):
        return dice_loss(pred, target, self.num_classes, self.ignore_idx)
  
  
## focal loss
def _label_to_onehot(target:torch.Tensor, num_classes:int, ignore_idx):
    """onehot encoding for 1 channel labelmap

    Args:
        target (torch.Tensor): shape (N, 1, H, W) have label values
        num_classes (int): the number of classes
        ignore_idx (int): the index which is ignored
    """
    onehot = torch.zeros((target.shape[0], num_classes, target.shape[1], target.shape[2]), dtype=torch.float64)
    for c in range(num_classes):
        if c in ignore_idx:
          continue
        onehot[:, c, :, :] += (target==c)
    return onehot
     
    
def focal_loss(pred:torch.Tensor, target:torch.Tensor, alpha, gamma, num_classes, ignore_idx=None, reduction="sum"):
    assert pred.shape[0] == target.shape[0],\
        "pred tensor and target tensor must have same batch size"
    
    if num_classes == 1:
        pred = F.sigmoid(pred)
    
    else:
        pred = F.softmax(pred, dim=1).float()

    onehot = _label_to_onehot(target, num_classes, ignore_idx)
    focal_loss = 0

    focal = torch.pow((1-pred), gamma) # (B, C, H, W)
    ce = -torch.log(pred) # (B, C, H, W)
    focal_loss = alpha * focal * ce * onehot
    focal_loss = torch.sum(focal_loss, dim=1) # (B, H, W)
    
    if reduction == 'none':
        # loss : (B, H, W)
        loss = focal_loss
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(focal_loss)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(focal_loss)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    
    return loss
    

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha, gamma, ignore_idx=None, reduction='sum'):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        if self.num_classes == 1:
            pred = F.sigmoid(pred)
        else:
            pred = F.softmax(pred, dim=1).float()
        
        return focal_loss(pred, target, self.alpha, self.gamma, self.num_classes, self.ignore_idx, self.reduction)