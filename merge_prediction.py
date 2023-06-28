import os
import csv
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def load_full_images_csv(path):
    with open(path, newline="") as csvfile:

        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        images = []
        for row in spamreader:
            images.append(row[0])
            
        return images[1:]


def load_patches_csv(path):
    with open(path, newline="") as csvfile:

        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        images = []
        for row in spamreader:
            images.append(row[0])
            
        return images[1:]
    
 
def merge(img_dict, filename, patches_path, results_path, img_size=384):    
    j = [int(img_dict[filename][n].split("_")[2]) for n in range(len(img_dict[filename]))]
    k = [int(img_dict[filename][n].split("_")[4]) for n in range(len(img_dict[filename]))]
    x_size = len(np.unique(j))
    y_size = len(np.unique(k))
    
    result = np.zeros((x_size * img_size, y_size * img_size))

    for x in tqdm(range(x_size), desc=f"Merging {filename}"):
        for y in range(y_size):
            try:
                # Example:
                # validation/rgbn_patch_4_1_by_4_LC08_L1TP_003052_20160120_20170405_01_T1.npy
                patch_name = os.path.join(patches_path, f"*_{x + 1}_by_{y + 1}_{filename}.npy")
                file_path = glob.glob(patch_name)[0]
                gt_patch = np.load(file_path)
                
            except Exception:
                continue

            result[x*img_size : x*img_size + img_size, y*img_size : y*img_size + img_size] = gt_patch
    
    result = result.astype(np.uint8) * 255
    im = Image.fromarray(result)
    name = os.path.join(results_path, f"{filename}.jpeg")
    im.save(name)
    print(f"Saved {filename}")
            
#TODO trzeba wyłuskać te współrzedne ktore sa jakies dziwne 
# zrobic array o wymiarach z tych wspolrzednych 
# i na kazde miejsce w tym array wstawic array z całym obrazkiem


# i jeszcze ogarnac wartosci tych metryk
#architekture z resnet fote wstawic
# sprawdzić na gicie najnowszy kod i hyperparameters stamtad

#potem skleic duze obrazki w zestawienie w prezce

#napisac wnioski

#przejrzec kod zeby wiedzeic co to jest za projekt

# wstawic jakies info do prezentacji (nie wiem jakis kod najlepszego modelu albo cokolwiek )            
            
if __name__ == "__main__":
    patches_path = "./results/validation"
    results_path = f"./results/validation_merge"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    img_names = load_full_images_csv("./data/archive/38-Cloud_test/test_sceneids_38-Cloud.csv")
    patch_names = load_patches_csv("./data/archive/38-Cloud_test/test_patches_38-Cloud.csv")
    
    names = dict()
    
    for i in img_names:
        for p in patch_names:
            if i in p:
                names[i] = (names.get(i, [])) + [p]       
                
    for i in img_names:
        names[i] = sorted(names[i])
    
    # Merging patches to full info
    for img in img_names:
        merge(names, img, patches_path, results_path)   
        