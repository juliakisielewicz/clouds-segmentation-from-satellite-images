import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from dataset import Clouds
from unet import CloudUnet
from attunet import Attunet
from resunet import ResUnet
from transformer import SegFormer


def plot_comparison(input_images, pred_segmentations, gt_segmentations, idx, folder_name, grid_len=4):
    fig, axes = plt.subplots(grid_len, 3, figsize=(12, 4 * grid_len))
    
    for i, (input_image, pred_segmentation, gt_segmentation) in enumerate(zip(input_images, pred_segmentations, gt_segmentations)):
        img = np.transpose(input_image[:3], (1, 2, 0))
        img[0] = (img[0] - np.min(img[0])) / (np.max(img[0]) - np.min(img[0]) + 0.000001)
        img[1] = (img[1] - np.min(img[1])) / (np.max(img[1]) - np.min(img[1]) + 0.000001)
        img[2] = (img[2] - np.min(img[2])) / (np.max(img[2]) - np.min(img[2]) + 0.000001)

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_segmentation, cmap="gray")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(gt_segmentation, cmap="gray")
        axes[i, 2].axis("off")

    axes[0, 0].set_title("Input Image", fontsize=20)
    axes[0, 1].set_title("Predicted Segmentation", fontsize=20)
    axes[0, 2].set_title("Ground Truth Segmentation", fontsize=20)
    
    plt.tight_layout()
    fig.savefig(f"results/{folder_name}/segmentation_example_{idx}.png", bbox_inches="tight")
    plt.close()


def main():
    batch_size = 3
    nb_of_grid_plots = 10
    folder_name = "cloud-unet-final"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cloud_dataset = Clouds("./data/stacked", "./data/gt", augment=False)
    _, validation_dataset = random_split(cloud_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(420))
    
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    model = CloudUnet().to(device)
    # model = Attunet(img_ch=4, output_ch=2).to(device)
    # model = ResUnet(in_channels=4, num_classes=2).to(device)
    
    folder_path = f"results/{folder_name}"
    print("Checkpoint:", sorted([i for i in os.listdir(folder_path) if '.pt' in i])[-1])
    checkpoint = torch.load(f"{folder_path}/{sorted([i for i in os.listdir(folder_path) if '.pt' in i])[-1]}")
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    normalize = T.Normalize([0.2007, 0.1983, 0.2110, 0.2350], 
                            [0.0658, 0.0631, 0.0661, 0.0673])
    
    with torch.no_grad():
        for idx, (input_img, gt) in tqdm(enumerate(validation_dataloader), desc="Creating plots", total=nb_of_grid_plots):
            input_img = input_img.to(device)
            gt = gt.to(device)
            
            input_img_norm = normalize(input_img)
            predicted_mask = model(input_img_norm)
            predicted_mask = torch.argmax(predicted_mask, dim=1).squeeze().cpu().numpy()
            
            input_img = input_img.squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()
            
            plot_comparison(input_img, predicted_mask, gt, idx, folder_name, grid_len=batch_size)
            
            if idx >= nb_of_grid_plots:
                break


if __name__ == "__main__":
    main()