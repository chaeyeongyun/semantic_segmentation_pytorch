import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import List

###### dice loss
###### dice loss
def dice_coefficient(pred:torch.Tensor, target:torch.Tensor, num_classes:int):
    """calculate dice coefficient

    Args:
        pred (torch.Tensor): (N, num_classes, H, W)
        target (torch.Tensor): (N, H, W)
        num_classes (int): the number of classes
    """
    
    if num_classes == 1:
        target = target.type(pred.type())
        pred = torch.sigmoid(pred)
        # target is onehot label
    else:
        target = target.type(pred.type()) # target과 pred의 type을 같게 만들어준다.
        target = torch.eye(num_classes)[target.long()].to(pred.device) # (N, H, W, num_classes)
        target = target.permute(0, 3, 1, 2) # (N, num_classes, H, W)
        pred = F.softmax(pred, dim=1)
    
    inter = torch.sum(pred*target, dim=(2, 3)) # (N, num_classes)
    sum_sets = torch.sum(pred+target, dim=(2, 3)) # (N, num_classes)
    dice_coefficient = (2*inter / (sum_sets+1e-6)).mean(dim=0) # (num_classes)
    return dice_coefficient
        
        
def dice_loss(pred, target, num_classes, weights:tuple=None):
    """_summary_

    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
        ignore_idx (_type_, optional): _description_. Defaults to None.
        weights(tuple) : the weights to apply to each class
    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    if not isinstance(pred, torch.Tensor) :
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(pred)}")

    dice = dice_coefficient(pred, target, num_classes)
    if weights is not None:
        dice_loss = 1-dice
        weights = torch.Tensor(weights)
        dice_loss = dice_loss * weights
        dice_loss = dice_loss.mean()
        
    else: 
        dice = dice.mean()
        dice_loss = 1 - dice
        
    return dice_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weights:tuple=None):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights
    def forward(self, pred, target):
        return dice_loss(pred, target, self.num_classes, weights=self.weights)

  
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
