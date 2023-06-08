import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import Clouds
from unet import CloudUnet
from


def plot_comparison(input_image, predicted_segmentation, ground_truth_segmentation):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
    img = np.transpose(input_image, (1, 2, 0))
    img[0] = (img[0] - np.min(img[0])) / (np.max(img[0]) - np.min(img[0]) + 0.000001)
    img[1] = (img[1] - np.min(img[1])) / (np.max(img[1]) - np.min(img[1]) + 0.000001)
    img[2] = (img[2] - np.min(img[2])) / (np.max(img[2]) - np.min(img[2]) + 0.000001)

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(predicted_segmentation, cmap="gray")
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    axes[2].imshow(ground_truth_segmentation, cmap="gray")
    axes[2].set_title("Ground Truth Segmentation")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def load_img(path):
    rgbn_ar = np.load(path)
    result = rgbn_ar[:3].astype(np.float64)
    return result


def main():    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cloud_dataset = Clouds("./data/stacked", "./data/gt")
    cloud_model = CloudUnet().to(device)
    
    checkpoint = torch.load("results/Cloud_UNet/Cloud_UNet_epoch_30.pt")
    cloud_model.load_state_dict(checkpoint)
    cloud_model.eval()
    
    with torch.no_grad():
        for input_img, gt in cloud_dataset:
            input_img = input_img[None].to(device)
            gt = gt[None].to(device)
            
            predicted_mask = cloud_model(input_img)
            
            predicted_mask = torch.argmax(predicted_mask, dim=1).squeeze().cpu().numpy()
            input_img = input_img.squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()
            
            plot_comparison(input_img[:3], gt, predicted_mask)


if __name__ == "__main__":
    main()