import os
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn


class Clouds(Dataset):
    def __init__(self, stacked_path, gt_path, augment=True):
        super(Clouds, self).__init__()
        self.stacked_files = sorted([os.path.join(stacked_path, file) for file in os.listdir(stacked_path)])
        self.gt_files = sorted([os.path.join(gt_path, file) for file in os.listdir(gt_path)])
        
        # Precomputed
        self.mean = [0.2007, 0.1983, 0.2110, 0.2350]
        self.std = [0.0658, 0.0631, 0.0661, 0.0673]
        
        # Augmentation:
        self.augment = augment
        self.normalize = T.Normalize(self.mean, self.std)
        self.transforms = T.Compose([
            T.RandomRotation((-30, 30), interpolation=T.InterpolationMode.NEAREST),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])

    def __len__(self):
        return len(self.stacked_files)
    
    def __getitem__(self, idx):        
        rgbn_img = torch.from_numpy(np.load(self.stacked_files[idx])).float()
        gt_img = torch.from_numpy(np.load(self.gt_files[idx])).float()
        
        if self.augment:
            state = torch.get_rng_state()
            rgbn_img = self.transforms(rgbn_img)
            rgbn_img = self.normalize(rgbn_img)
            torch.set_rng_state(state)
            gt_img = self.transforms(gt_img.unsqueeze(0)).squeeze()
        
        return rgbn_img, gt_img
        