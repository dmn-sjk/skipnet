from torchvision import datasets, transforms
import torch
from datasets.augmentations import get_augs
from utils.utils import worker_init_fn


def get_cifar_100_dataloader(data_root, split, bs, num_workers=4, pin_memory=True, normalize=True):
    
    dataset = datasets.CIFAR100(root=data_root,
                                train=True if split == 'train' else False,
                                download=True,
                                transform=get_augs('cifar_10', split, normalize))

    return torch.utils.data.DataLoader(dataset,
        batch_size=bs, shuffle=True if split == 'train' else False, 
        sampler=None, num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=worker_init_fn)

def get_cifar_10_dataloader(data_root, split, bs, num_workers=4, pin_memory=True, normalize=True):
    
    dataset = datasets.CIFAR10(root=data_root,
                                train=True if split == 'train' else False,
                                download=True,
                                transform=get_augs('cifar_10', split, normalize))

    return torch.utils.data.DataLoader(dataset,
        batch_size=bs, shuffle=True if split == 'train' else False, 
        sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
        worker_init_fn=worker_init_fn)

    