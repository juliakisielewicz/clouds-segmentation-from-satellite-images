import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def accuracy(predicted, gt):
    intersect = np.sum(predicted * gt)
    union = np.sum(predicted) + np.sum(gt) - intersect
    xor = np.sum(gt == predicted)
    acc = np.mean(xor / (union + xor - intersect))
    return round(acc, 5)
    
    
def iou(predicted, gt):
    intersection = np.logical_and(gt, predicted)
    union = np.logical_or(gt, predicted)
    
    if not np.any(union):
        return 0
    
    iou_score = np.sum(intersection) / np.sum(union)
    return round(iou_score, 5)


def save_plots(results_path, train_loss, val_loss, train_accuracy, val_accuracy, train_iou, val_iou):
    epochs = list(range(len(train_loss)))
    plt.plot(epochs, train_loss, color="teal", label="Training loss")
    plt.plot(epochs, val_loss, color="crimson", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(results_path, "loss_plot.png"), bbox_inches="tight")
    plt.close()
    
    plt.plot(epochs, train_accuracy, color="teal", label="Training accuracy")
    plt.plot(epochs, val_accuracy, color="crimson", label="Validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_path, "accuracy_plot.png"), bbox_inches="tight")
    plt.close()
    
    plt.plot(epochs, train_iou, color="teal", label="Training IOU")
    plt.plot(epochs, val_iou, color="crimson", label="Validation IOU")
    plt.title("Training and Validation IOU")
    plt.xlabel("Epochs")
    plt.ylabel("IOU")
    plt.legend()
    plt.savefig(os.path.join(results_path, "iou_plot.png"), bbox_inches="tight")
    plt.close()


def merge_files():    
    blue_files = os.listdir("./data/patches/train_blue")
    Path("./data/stacked").mkdir(parents=True, exist_ok=True)
    Path("./data/gt").mkdir(parents=True, exist_ok=True)
        
    for i, blue_file in enumerate(tqdm(blue_files, desc="Merging files")):
        file = blue_file[4:]
        green_file = os.path.join("./data/patches/train_green", ("green" + file))
        red_file = os.path.join("./data/patches/train_red", ("red" + file))
        blue_file = os.path.join("./data/patches/train_blue", ("blue" + file))
        nir_file = os.path.join("./data/patches/train_nir", ("nir" + file))
        gt_file = os.path.join("./data/patches/train_gt", ("gt" + file))
        
        red_img = np.array(Image.open(red_file))
        green_img = np.array(Image.open(green_file))
        blue_img = np.array(Image.open(blue_file))
        nir_img = np.array(Image.open(nir_file))
        gt_img = np.array(Image.open(gt_file))

        rgbn_img = np.stack([red_img, green_img, blue_img, nir_img])
        
        rgbn_img = np.array(rgbn_img / 65536).astype(np.float16)
        gt_img = np.array(gt_img / 255).astype(np.float16)
        
        # FIXME: temporary
        if i >= 20000:
            break
        
        # skipping completely black images
        if not np.any(rgbn_img):
            continue
        
        np.save(f"./data/stacked/rgbn_{file[1:-4]}.npy", rgbn_img)
        np.save(f"./data/gt/gt_{file[1:-4]}.npy", gt_img)


if __name__ == "__main__":
    merge_files()
