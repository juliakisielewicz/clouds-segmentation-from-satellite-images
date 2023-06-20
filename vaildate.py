import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import Clouds
from unet import CloudUnet
from attunet import Attunet
from resunet import ResUnet


def merge_files():
    blue_files = os.listdir("./data/archive(2)/38-Cloud_test/test_blue")
    Path("./data/stacked_validation").mkdir(parents=True, exist_ok=True)
        
    for blue_file in tqdm(blue_files, desc="Merging files"):
        file = blue_file[4:]
        green_file = os.path.join("./data/archive(2)/38-Cloud_test/test_green", ("green" + file))
        red_file = os.path.join("./data/archive(2)/38-Cloud_test/test_red", ("red" + file))
        blue_file = os.path.join("./data/archive(2)/38-Cloud_test/test_blue", ("blue" + file))
        nir_file = os.path.join("./data/archive(2)/38-Cloud_test/test_nir", ("nir" + file))
        
        red_img = np.array(Image.open(red_file))
        green_img = np.array(Image.open(green_file))
        blue_img = np.array(Image.open(blue_file))
        nir_img = np.array(Image.open(nir_file))

        rgbn_img = np.stack([red_img, green_img, blue_img, nir_img])
        
        #normalization
        rgbn_img = np.array(rgbn_img / 65536).astype(np.float16)
        
        # skipping completely black images
        if not np.any(rgbn_img):
            continue
        
        np.save(f"./data/stacked_validation/rgbn_{file[1:-4]}.npy", rgbn_img)
        
        
class CloudsVal(Dataset):
    def __init__(self, stacked_path):
        super(CloudsVal, self).__init__()
        self.stacked_files = sorted([os.path.join(stacked_path, file) for file in os.listdir(stacked_path)])
        
        # Precomputed
        self.mean = [0.2007, 0.1983, 0.2110, 0.2350]
        self.std = [0.0658, 0.0631, 0.0661, 0.0673]

        self.normalize = T.Normalize(self.mean, self.std)

    def __len__(self):
        return len(self.stacked_files)
    
    def __getitem__(self, idx):        
        rgbn_img = torch.from_numpy(np.load(self.stacked_files[idx])).float()
        img_name = self.stacked_files[idx].split("\\")[-1]
        rgbn_img = self.normalize(rgbn_img)
        
        return rgbn_img, img_name
        
        

def main():
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device: {device}")
    
    results_path = f"./results/validation"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    cloud_dataset = CloudsVal("./data/stacked_validation")
    dataloader = DataLoader(cloud_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    
    model = ResUnet(in_channels=4, num_classes=2).to(device)
    checkpoint = torch.load("./results/cloud-res-unet/cloud-res-unet_epoch_20.pt")
    model.load_state_dict(checkpoint)
    
    model.eval()
        
    with torch.no_grad():    
        for stacked_img, img_names in tqdm(dataloader, desc="Predicting"):
            stacked_img = stacked_img.to(device)
            
            output = model(stacked_img)
            predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            for img, name in zip(predicted_mask, img_names):
                np.save(f"{results_path}/{name}", img)


if __name__ == "__main__":
    main()