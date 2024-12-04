
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

from datasets.augmentations import get_augs


DOMAINS = ['RealWorld', 'Art', 'Clipart', 'Product']
        
def get_office_home_dataloader(data_root, split, bs, num_workers=4, domain='RealWorld'):
    files_list_path = os.path.join(data_root, 'OfficeHome', f'{domain}_list.txt')
 
    with open(files_list_path, 'r') as f:
        files = f.readlines()

    if split == 'train':
        tr_size = int(0.8 * len(files))
        files, _ = torch.utils.data.random_split(files, [tr_size, len(files) - tr_size])
    elif split == 'val':
        tr_size = int(0.8 * len(files))
        _, files = torch.utils.data.random_split(files, [tr_size, len(files) - tr_size])
    elif split == 'all':
        print("Taking all the files form OfficeHome dataset")
        pass
    else:
        raise NotImplementedError

    dataset = ImageList(files, transform=get_augs('office_home', split))
    dataset.classes = list(range(65))
    
    return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=False)

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)