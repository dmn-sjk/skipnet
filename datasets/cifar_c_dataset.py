"""
https://robustbench.github.io/
"""

import os
from typing import Callable, Dict, Optional, Sequence, Tuple, Set
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from pathlib import Path
from enum import Enum

from utils.zenodo_download import zenodo_download, DownloadError
from datasets.augmentations import get_augs
from utils.utils import worker_init_fn


class BenchmarkDataset(Enum):
    cifar_10 = 'cifar10'
    cifar_100 = 'cifar_100'

CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

ZENODO_CORRUPTIONS_LINKS: Dict[BenchmarkDataset, Tuple[str, Set[str]]] = {
    BenchmarkDataset.cifar_10: ("2535967", {"CIFAR-10-C.tar"}),
    BenchmarkDataset.cifar_100: ("3555552", {"CIFAR-100-C.tar"})
}

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.cifar_10: "CIFAR-10-C",
    BenchmarkDataset.cifar_100: "CIFAR-100-C",
}

TRAIN_IDXS_PATH = 'datasets/cifar_c_val_train/train.npy'
VAL_IDXS_PATH = 'datasets/cifar_c_val_train/val.npy'


def get_cifar_10_c_dataloader(data_root, split, bs, corruption='gaussian_noise', severity=5, 
                              num_workers=4, pin_memory=True, normalize=True):
    dataset = create_cifarc_dataset(dataset_name='cifar10_c',
                                    severity=severity,
                                    data_dir=data_root,
                                    corruption=corruption,
                                    transform=get_augs('cifar10_c', split, normalize)) 

    if split == 'train':
        idxs = np.load(TRAIN_IDXS_PATH).tolist()
        dataset.samples = np.array(dataset.samples, dtype=object)[idxs].tolist()
    elif split == 'val':
        idxs = np.load(VAL_IDXS_PATH).tolist()
        dataset.samples = np.array(dataset.samples, dtype=object)[idxs].tolist()
    elif split == 'all':
        pass
    else:
        raise NotImplementedError
    
    dataset.classes = list(range(10))
    
    return torch.utils.data.DataLoader(dataset,
        batch_size=bs, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)

def get_cifar_100_c_dataloader(data_root, corruption, severity, split, bs, 
                              num_workers=4, pin_memory=True, normalize=True):
    dataset = create_cifarc_dataset(dataset_name='cifar100_c',
                                    severity=severity,
                                    data_dir=data_root,
                                    corruption=corruption,
                                    transform=get_augs('cifar100_c', split, normalize))
    
    if split == 'train':
        idxs = np.load(TRAIN_IDXS_PATH).tolist()
        dataset.samples = np.array(dataset.samples, dtype=object)[idxs].tolist()
    elif split == 'val':
        idxs = np.load(VAL_IDXS_PATH).tolist()
        dataset.samples = np.array(dataset.samples, dtype=object)[idxs].tolist()
    elif split == 'all':
        pass
    else:
        raise NotImplementedError

    dataset.classes = list(range(100))
    
    return torch.utils.data.DataLoader(dataset,
        batch_size=bs, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)



class CustomCifarDataset(data.Dataset):
    def __init__(self, samples, transform=None):
        super(CustomCifarDataset, self).__init__()

        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        img, label, domain = self.samples[index]
        if self.transform is not None:
            img = Image.fromarray(np.uint8(img * 255.)).convert('RGB')
            img = self.transform(img)
        else:
            img = torch.tensor(img.transpose((2, 0, 1)))

        return img, torch.tensor(label).long() # , domain

    def __len__(self):
        return len(self.samples)

def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)

def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0, worker_init_fn=worker_init_fn)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_corruptions_cifar(
        dataset: BenchmarkDataset,
        n_examples: int = 10000,
        severity: int = 5,
        data_dir: str = './data',
        corruptions: Sequence[str] = CORRUPTIONS,
        shuffle: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset]

    if not data_root_dir.exists():
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / 'labels.npy'
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')
        if not corruption_file_path.is_file():
            raise DownloadError(
                f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test

def load_cifar10c(
        n_examples: int = 10000,
        severity: int = 5,
        data_dir: str = './data',
        shuffle: bool = False,
        corruptions: Sequence[str] = CORRUPTIONS,
        prepr: Optional[str] = 'none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(BenchmarkDataset.cifar_10, n_examples,
                                  severity, data_dir, corruptions, shuffle)


def load_cifar100c(
        n_examples: int = 10000,
        severity: int = 5,
        data_dir: str = './data',
        shuffle: bool = False,
        corruptions: Sequence[str] = CORRUPTIONS,
        prepr: Optional[str] = 'none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(BenchmarkDataset.cifar_100, n_examples,
                                  severity, data_dir, corruptions, shuffle)
