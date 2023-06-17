import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchmetrics import Dice, JaccardIndex, Accuracy

from dataset import Clouds
from unet import CloudUnet
from attunet import Attunet
from resunet import ResUnet
from utils import save_plots

torch.backends.cudnn.benchmark = True


def train(name, epochs, batch_size, learning_rate):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device: {device}")
    
    results_path = f"./results/{name}"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    print("Initializing dataset")
    cloud_dataset = Clouds("./data/stacked", "./data/gt")
    train_dataset, validation_dataset = random_split(cloud_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(420))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    # # Precomputing mean and std of whole dataset to normalize images
    # loader = DataLoader(cloud_dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    
    # mean = 0.0
    # for images, _ in tqdm(loader):
    #     batch_samples = images.size(0) 
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    # mean = mean / len(loader.dataset)

    # var = 0.0
    # pixel_count = 0
    # for images, _ in tqdm(loader):
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    #     pixel_count += images.nelement()
    # std = torch.sqrt(var / pixel_count)
    
    # print(mean)
    # print(std)
    
    # Different models to choose
    # model = CloudUnet().to(device)
    # model = SegFormer(
    #     in_channels=4,
    #     widths=[64, 128, 256, 512],
    #     depths=[3, 4, 6, 3],
    #     all_num_heads=[1, 2, 4, 8],
    #     patch_sizes=[7, 3, 3, 3],
    #     overlap_sizes=[4, 2, 2, 2],
    #     reduction_ratios=[8, 4, 2, 1],
    #     mlp_expansions=[4, 4, 4, 4],
    #     decoder_channels=256,
    #     scale_factors=[8, 4, 2, 1],
    #     num_classes=2,
    #     scale_factor=4
    # ).to(device)
    model = Attunet(img_ch=4, output_ch=2).to(device)
    # model = ResUnet(in_channels=4, num_classes=2).to(device)
    
    # FIXME: temp
    checkpoint = torch.load("results/cloud-attention-unet/cloud-attention-unet_epoch_7.pt")
    model.load_state_dict(checkpoint)
        
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 0.6]).to(device))
    jaccard = JaccardIndex(task="multiclass", num_classes=2).to(device)
    dice = Dice(average="macro", num_classes=2).to(device)
    accuracy = Accuracy(task="multiclass", num_classes=2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    
    train_loss = []
    validation_loss = []
    train_accuracy = []
    validation_accuracy = []
    train_dice = []
    validation_dice = []
    train_jaccard = []
    validation_jaccard = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Current learning rate: {round(scheduler.get_last_lr()[0], 6)}")
        
        running_loss = 0.0
        running_val_loss = 0.0
        running_accuracy = 0.0
        running_val_acccuracy = 0.0
        running_jaccard = 0.0
        running_val_jaccard = 0.0
        running_dice = 0.0
        running_val_dice = 0.0
        
        for stacked_img, gt_img in tqdm(train_dataloader, desc="Train"):            
            stacked_img = stacked_img.to(device)
            gt_img = gt_img.to(device).long()

            optim.zero_grad()
            output = model(stacked_img)
            loss = criterion(output, gt_img)
            loss.backward()
            optim.step()
            
            running_loss += loss.item()
            running_accuracy += accuracy(output, gt_img).cpu().numpy()
            running_dice += dice(output, gt_img).cpu().numpy()
            running_jaccard += jaccard(output, gt_img).cpu().numpy()
            
        model.eval()
           
        with torch.no_grad():    
            for stacked_img, gt_img in tqdm(validation_dataloader, desc="Validation"):
                stacked_img = stacked_img.to(device)
                gt_img = gt_img.to(device).long()
                
                output = model(stacked_img)
                loss = criterion(output, gt_img)
                
                running_val_loss += loss.item()
                running_val_acccuracy += accuracy(output, gt_img).cpu().numpy()
                running_val_dice += dice(output, gt_img).cpu().numpy()
                running_val_jaccard += jaccard(output, gt_img).cpu().numpy()
    
        model.train()
        scheduler.step()

        running_loss = running_loss / len(train_dataloader)
        running_val_loss = running_val_loss / len(validation_dataloader)
        running_accuracy = running_accuracy / len(train_dataloader)
        running_val_acccuracy = running_val_acccuracy / len(validation_dataloader)
        running_dice = running_dice / len(train_dataloader)
        running_val_dice = running_val_dice / len(validation_dataloader)
        running_jaccard = running_jaccard / len(train_dataloader)
        running_val_jaccard = running_val_jaccard / len(validation_dataloader)
        
        train_loss.append(running_loss)
        validation_loss.append(running_val_loss)
        train_accuracy.append(running_accuracy)
        validation_accuracy.append(running_val_acccuracy)
        train_dice.append(running_dice)
        validation_dice.append(running_val_dice)
        train_jaccard.append(running_jaccard)
        validation_jaccard.append(running_val_jaccard)
    
        print(f"loss = {round(running_loss, 5)}; validation loss = {round(running_val_loss, 5)}")
        print(f"accuracy = {round(running_accuracy, 5)}; validation accuracy = {round(running_val_acccuracy, 5)}")
        print(f"jaccard = {round(running_jaccard, 5)}; validation jaccard = {round(running_val_jaccard, 5)}")
        print(f"dice = {round(running_dice, 5)}; validation dice = {round(running_val_dice, 5)}\n")
        
        torch.save(model.state_dict(), os.path.join(results_path, f"{name}_epoch_{epoch + 1}.pt"))

    save_plots(results_path, train_loss, validation_loss, train_accuracy, validation_accuracy, train_dice, validation_dice, 
               train_jaccard, validation_jaccard)


if __name__ == "__main__":
    name = "cloud-attention-unet"
    epochs = 20
    batch_size = 2
    learning_rate = 5e-4
    train(name, epochs, batch_size, learning_rate)
