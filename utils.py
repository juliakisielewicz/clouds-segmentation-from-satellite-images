import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def save_plots(results_path, train_loss, val_loss, train_accuracy, val_accuracy, train_dice, val_dice, train_jaccard, val_jaccard):
    epochs = list(range(len(train_loss)))
    plt.plot(epochs, train_loss, color="teal", label="Training loss")
    plt.plot(epochs, val_loss, color="crimson", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(results_path, "loss_plot.png"), bbox_inches="tight")
    plt.close()
    
    plt.plot(epochs, train_accuracy, color="teal", label="Training accuracy")
    plt.plot(epochs, val_accuracy, color="crimson", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_path, "accuracy_plot.png"), bbox_inches="tight")
    plt.close()
    
    plt.plot(epochs, train_dice, color="teal", label="Training dice")
    plt.plot(epochs, val_dice, color="crimson", label="Validation dice")
    plt.title("Training and validation dice")
    plt.xlabel("Epochs")
    plt.ylabel("Dice")
    plt.legend()
    plt.savefig(os.path.join(results_path, "dice_plot.png"), bbox_inches="tight")
    plt.close()
    
    plt.plot(epochs, train_jaccard, color="teal", label="Training jaccard")
    plt.plot(epochs, val_jaccard, color="crimson", label="Validation jaccard")
    plt.title("Training and validation jaccard")
    plt.xlabel("Epochs")
    plt.ylabel("Jaccard")
    plt.legend()
    plt.savefig(os.path.join(results_path, "jaccard_plot.png"), bbox_inches="tight")
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
        
        #normalization
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
