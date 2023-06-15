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
        
        # for s, g in zip(self.stacked_files[:20], self.gt_files[:20]):
        #     print(s, g)
        
        # self.stacked_files = self.stacked_files[:100]
        # self.gt_files = self.gt_files[:100]
        
        # imgs = torch.stack([torch.from_numpy(np.load(i)).float() for i in self.stacked_files])
        # self.mean = imgs.view(4, -1).mean(dim=1)
        # self.std = imgs.view(4, -1).std(dim=1)
        
        # # Augmentation: 
        # self.transforms = T.Compose([
        #     T.Normalize((self.mean), (self.std))
        #     # T.RandomRotation((-180, 180)),
        #     # T.RandomHorizontalFlip(),
        #     # T.RandomVerticalFlip()
        # ])

    def __len__(self):
        return len(self.stacked_files)
    
    def __getitem__(self, idx):        
        rgbn_img = torch.from_numpy(np.load(self.stacked_files[idx])).float()
        gt_img = torch.from_numpy(np.load(self.gt_files[idx])).float()
        
        # state = torch.get_rng_state()
        # rgbn_img = self.transforms(rgbn_img)
        # torch.set_rng_state(state)
        # gt_img = self.transforms(gt_img.unsqueeze(1)).squeeze()
        
        return rgbn_img, gt_img
        