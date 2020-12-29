import os

import cv2 as cv
import numpy as np
from numpy.lib.function_base import append
import torch
import torchvision.transforms as transforms

disease_labels = {"PM": 0, "AMD": 1, "PCV": 2, "DME": 3, "NM": 4}

train_to_val = 4


def traverse_dataset(img_root_dir):
    img_paths = {"train": [], "val": []}
    img_labels = {"train": [], "val": []}
    # sub_dir_name is also disease name, e.g. AMD
    for sub_dir_name in os.listdir(img_root_dir):
        sub_dir = os.path.join(img_root_dir, sub_dir_name)
        sub_sub_dir_names = os.listdir(sub_dir)
        # sub_sub_dir_name is also case name, e.g. A-0001
        # add img to train first
        for sub_sub_dir_name in sub_sub_dir_names[:len(sub_sub_dir_names)*train_to_val//(train_to_val+1)]:
            sub_sub_dir = os.path.join(sub_dir, sub_sub_dir_name)
            img_paths["train"].append(sub_sub_dir)
            img_labels["train"].append(disease_labels[sub_dir_name])
        # then to val
        for sub_sub_dir_name in sub_sub_dir_names[len(sub_sub_dir_names)*train_to_val//(train_to_val+1):]:
            sub_sub_dir = os.path.join(sub_dir, sub_sub_dir_name)
            img_paths["val"].append(sub_sub_dir)
            img_labels["val"].append(disease_labels[sub_dir_name])
    return img_paths, img_labels


class RetinalOCT(torch.utils.data.Dataset):
    def __init__(self, img_path, img_label, data_transform):
        self.img_path = img_path
        self.img_label = img_label
        self.data_transform = data_transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        dir_path = self.img_path[item]
        imgs = []
        idx = 0
        appenddix = ''
        for i, file_name in enumerate(os.listdir(dir_path)):
            idx = i
            file_path = os.path.join(dir_path, file_name)
            appenddix = file_path
            imgs.append(cv.resize(cv.imread(file_path, flags=cv.IMREAD_GRAYSCALE), (224,224)))

        if idx < 18: # 如果该病人不足19张图片,那么idx将会小于18
            for i in range(18 - idx): # 将该病人的最后一张图片多次添加，直到凑足19张
                imgs.append(cv.resize(cv.imread(appenddix, flags=cv.IMREAD_GRAYSCALE), (224,224)))
        return self.data_transform(np.stack(imgs,axis=2)), self.img_label[item]

def Data_loader_3D(img_root_dir, batch_size):
    img_paths, img_labels = traverse_dataset(img_root_dir)
    data_transforms = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ),
                                 (0.5, ))
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ),
                                 (0.5, ))
        ]),
    }
    image_datasets = {x: RetinalOCT(img_paths[x], img_labels[x], data_transforms[x])
                      for x in ["train", "val"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ["train", "val"]}

    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['train', 'val']}
    print(dataset_sizes)
    return image_datasets, dataloaders
