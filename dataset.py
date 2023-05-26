import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Clouds(Dataset):
    def __init__(self, stacked_path, gt_path):
        super(Clouds, self).__init__()
        self.stacked_files = sorted([os.path.join(stacked_path, file) for file in os.listdir(stacked_path)])
        self.gt_files = sorted([os.path.join(gt_path, file) for file in os.listdir(gt_path)])

    def __len__(self):
        return len(self.stacked_files)
    
    def __getitem__(self, idx):
        rgbn_img = torch.from_numpy(np.load(self.stacked_files[idx])).float()
        gt_img = torch.from_numpy(np.load(self.gt_files[idx])).float()
        
        return rgbn_img, gt_img
        