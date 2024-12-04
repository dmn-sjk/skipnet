from torchvision import transforms


NORMS = {
    'cifar': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]},
    # 'cifar_100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
    'svhn': {'mean': [0.4524,  0.4525,  0.4690], 'std': [0.2194,  0.2266,  0.2285]},
    'office_home': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    }


def get_augs(dataset, split, normalize=True):
    if 'cifar' in dataset:
        return _get_cifar_augs(dataset, split, normalize=normalize)
    elif 'office_home' in dataset:
        return _get_office_home_augs(split, normalize=normalize)
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
            mean=NORMS['cifar']['mean'], 
            std=NORMS['cifar']['std']
            )
        trans.append(norm)
 
    return transforms.Compose(trans)

def _get_office_home_augs(split, normalize=True, resize_size=256, crop_size=224):
    trans = []
    if split == 'train':
        trans.extend([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:    
        trans.extend([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])

    if normalize:
        norm = transforms.Normalize(
            mean=NORMS['office_home']['mean'], 
            std=NORMS['office_home']['std']
            )
        trans.append(norm)
 
    return transforms.Compose(trans)
 