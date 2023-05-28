import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torchsummary import summary
from torch.utils.data import DataLoader, random_split

from dataset import Clouds
from model import CloudUnet
from utils import save_plots, accuracy, iou

torch.backends.cudnn.benchmark = True

def train(name, epochs, batch_size, learning_rate):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device: {device}")
    
    results_path = f"./results/{name}"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    print("Initializing dataset")
    cloud_dataset = Clouds("./data/stacked", "./data/gt")
    train_dataset, validation_dataset = random_split(cloud_dataset, [0.8, 0.2])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    print("Cloud model UNet:")
    cloud_model = CloudUnet().to(device)
    print(summary(cloud_model, (4, 384, 384)))
    print("\n")
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.44, 0.56]).to(device))
    optim = torch.optim.AdamW(cloud_model.parameters(), lr=learning_rate)
    
    train_loss = []
    validation_loss = []
    train_accuracy = []
    validation_accuracy = []
    train_iou = []
    validation_iou = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        running_loss = 0.0
        running_val_loss = 0.0
        running_accuracy = 0.0
        running_val_acccuracy = 0.0
        running_iou = 0.0
        running_val_iou = 0.0
        
        for stacked_img, gt_img in tqdm(train_dataloader, desc="Train"):            
            stacked_img = stacked_img.to(device)
            gt_img = gt_img.to(device).long()

            optim.zero_grad()
            output = cloud_model(stacked_img)
            loss = criterion(output, gt_img)
            loss.backward()
            optim.step()
            
            running_loss += loss.item()
            metrics_output = torch.argmax(output, dim=1).cpu().detach().numpy()
            metrics_gt = gt_img.cpu().detach().numpy()
            running_accuracy += accuracy(metrics_output, metrics_gt)
            running_iou += iou(metrics_output, metrics_gt)
            
        cloud_model.eval()
        
        with torch.no_grad():    
            for stacked_img, gt_img in tqdm(validation_dataloader, desc="Validation"):
                stacked_img = stacked_img.to(device)
                gt_img = gt_img.to(device).long()
                
                output = cloud_model(stacked_img)
                loss = criterion(output, gt_img)
                
                running_val_loss += loss.item()
                metrics_output = torch.argmax(output, dim=1).cpu().detach().numpy()
                metrics_gt = gt_img.cpu().detach().numpy()
                running_val_acccuracy += accuracy(metrics_output, metrics_gt)
                running_val_iou += iou(metrics_output, metrics_gt)
    
        cloud_model.train()

        running_loss = running_loss / len(train_dataloader)
        running_val_loss = running_val_loss / len(validation_dataloader)
        running_accuracy = running_accuracy / len(train_dataloader)
        running_val_acccuracy = running_val_acccuracy / len(validation_dataloader)
        running_iou = running_iou / len(train_dataloader)
        running_val_iou = running_val_iou / len(validation_dataloader)
        
        train_loss.append(running_loss)
        validation_loss.append(running_val_loss)
        train_accuracy.append(running_accuracy)
        validation_accuracy.append(running_val_acccuracy)
        train_iou.append(running_iou)
        validation_iou.append(running_val_iou)
    
        print(f"loss = {round(running_loss, 5)}; validation loss = {round(running_val_loss, 5)}")
        print(f"accuracy = {round(running_accuracy, 5)}; validation accuracy = {round(running_val_acccuracy, 5)}")
        print(f"IOU = {round(running_iou, 5)}; validation IOU = {round(running_val_iou, 5)}\n")
        
        torch.save(cloud_model.state_dict(), os.path.join(results_path, f"{name}_epoch_{epoch + 1}.pt"))

    save_plots(results_path, train_loss, validation_loss, train_accuracy, validation_accuracy, train_iou, validation_iou)


if __name__ == "__main__":
    name = "Cloud_UNet_2"
    epochs = 30
    batch_size = 4
    learning_rate = 1e-4
    train(name, epochs, batch_size, learning_rate)
