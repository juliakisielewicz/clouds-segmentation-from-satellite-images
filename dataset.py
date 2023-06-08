import os
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn


class Clouds(Dataset):
    def __init__(self, stacked_path, gt_path):
        super(Clouds, self).__init__()
        self.stacked_files = sorted([os.path.join(stacked_path, file) for file in os.listdir(stacked_path)])
        self.gt_files = sorted([os.path.join(gt_path, file) for file in os.listdir(gt_path)])
        
        # self.stacked_files = self.stacked_files[:100]
        # self.gt_files = self.gt_files[:100]
        
        self.transforms = T.RandomChoice([
            T.RandomRotation((-180, 180)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])

    def __len__(self):
        return len(self.stacked_files)
    
    def __getitem__(self, idx):        
        rgbn_img = torch.from_numpy(np.load(self.stacked_files[idx])).float()
        gt_img = torch.from_numpy(np.load(self.gt_files[idx])).float()
        
        state = torch.get_rng_state()
        rgbn_img = self.transforms(rgbn_img)
        torch.set_rng_state(state)
        gt_img = self.transforms(gt_img.unsqueeze(1)).squeeze()
        
        rgbn_img = fn.normalize(rgbn_img, mean=[torch.mean(rgbn_img)], std=[0.5])
        
        return rgbn_img, gt_img
        