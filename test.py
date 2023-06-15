import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import Clouds
from unet import CloudUnet
from transformer import SegFormer


def plot_comparison(input_image, predicted_segmentation, ground_truth_segmentation, idx):
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
    # plt.show()
    # fig.savefig(f"results/cloud-unet-2/segmentation_example_{idx}.png", bbox_inches="tight")
    fig.savefig(f"results/segformer/segmentation_example_{idx}.png", bbox_inches="tight")
    plt.close()


def main():    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cloud_dataset = Clouds("./data/stacked", "./data/gt")
    
    # model = CloudUnet().to(device)
    # checkpoint = torch.load("results/cloud-unet-2/cloud-unet-2_epoch_30.pt")
    model = SegFormer(
        in_channels=4,
        widths=[64, 128, 256, 512],
        depths=[3, 4, 6, 3],
        all_num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        overlap_sizes=[4, 2, 2, 2],
        reduction_ratios=[8, 4, 2, 1],
        mlp_expansions=[4, 4, 4, 4],
        decoder_channels=256,
        scale_factors=[8, 4, 2, 1],
        num_classes=2,
        scale_factor=4
        ).to(device)
    checkpoint = torch.load("results/segformer/segformer_epoch_17.pt")
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    with torch.no_grad():
        for idx, (input_img, gt) in enumerate(cloud_dataset):
            input_img = input_img[None].to(device)
            gt = gt[None].to(device)
            
            predicted_mask = model(input_img)
            
            predicted_mask = torch.argmax(predicted_mask, dim=1).squeeze().cpu().numpy()
            input_img = input_img.squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()
            
            plot_comparison(input_img[:3], gt, predicted_mask, idx)
            
            if idx >= 20:
                break


if __name__ == "__main__":
    main()