from PIL import Image
import numpy as np
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SegDataset(Dataset):
    def __init__(self, data_dir, resize=512, inputresize=True, targetresize=False, transform=None, target_transform=None):
        self.img_dir = os.path.join(data_dir, 'input')
        self.mask_dir = os.path.join(data_dir, 'target')
        self.resize = resize
        self.inputresize = inputresize
        self.targetresize = targetresize
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform    
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.inputresize: image = image.resize((self.resize, self.resize), resample=Image.BILINEAR) 
        image = TF.to_tensor(image)
        
        mask_path = os.path.join(self.mask_dir, filename)
        mask = Image.open(mask_path).convert('L') # size : (W, H), grayscale image
        if self.targetresize: mask = mask.resize((self.resize, self.resize), resample=Image.NEAREST)
        mask = np.array(mask) # (H, W)
        mask = torch.from_numpy(mask)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask, filename