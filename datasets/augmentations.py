from torchvision import transforms


NORMS_CIFAR = {
    'cifar': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]},
    # 'cifar_100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
    'svhn': {'mean': [0.4524,  0.4525,  0.4690], 'std': [0.2194,  0.2266,  0.2285]},
    }


def get_augs(dataset, split, normalize=True):
    if 'cifar' in dataset:
        return _get_cifar_augs(dataset, split, normalize=normalize)
    else:
        raise NotImplementedError
    

def _get_cifar_augs(dataset, split, normalize=True, resize=32):
    trans = []
    if split == 'train':
        trans.extend([
            transforms.RandomCrop(resize, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:    
        trans.extend([
            transforms.ToTensor(),
        ])

    if normalize:
        norm = transforms.Normalize(
            mean=NORMS_CIFAR['cifar']['mean'], 
            std=NORMS_CIFAR['cifar']['std']
            )
        trans.append(norm)
 
    return transforms.Compose(trans)