from datasets.cifar_dataset import get_cifar_100_dataloader, get_cifar_10_dataloader
from datasets.cifar_c_dataset import get_cifar_10_c_dataloader, get_cifar_100_c_dataloader
from datasets.office_home_dataset import get_office_home_dataloader
from datasets.cifar_c_dataset import CORRUPTIONS
from datasets.office_home_dataset import DOMAINS
from torch.utils.data import ConcatDataset, DataLoader


def get_dataloader(args, split):
    if args.dataset == 'cifar100':
        return get_cifar_100_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers)
    elif args.dataset == 'cifar10':
        return get_cifar_10_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers)
    elif args.dataset == 'cifar10c':
        return get_cifar_10_c_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers, 
                                        corruption=args.domain, severity=args.severity)
    elif args.dataset == 'cifar100c':
        return get_cifar_100_c_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers, 
                                          corruption=args.domain, severity=args.severity)
    elif args.dataset == 'cifar10c_all':
        loaders = []
        for corruption in CORRUPTIONS:
            loaders.append(get_cifar_10_c_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers, 
                                                     corruption=corruption, severity=args.severity))
        return merge_dataloaders(args, loaders)
            
    elif args.dataset == 'cifar100c_all':
        loaders = []
        for corruption in CORRUPTIONS:
            loaders.append(get_cifar_100_c_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers, 
                                                     corruption=corruption, severity=args.severity))
        return merge_dataloaders(args, loaders)

    elif args.dataset == 'office_home':
        return get_office_home_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers,
                                          domain=args.domain)
    elif args.dataset == 'office_home_all':
        loaders = []
        for domain in DOMAINS:
            loaders.append(get_office_home_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers,
                                                      domain=domain))
        return merge_dataloaders(args, loaders)
    else:
        raise NotImplementedError(args.dataset)
    
    
def merge_dataloaders(args, loaders):    
    # Concatenate the datasets from the input dataloaders
    datasets = [dataloader.dataset for dataloader in loaders]
    merged_dataset = ConcatDataset(datasets)

    # Create a new dataloader from the merged dataset
    return DataLoader(
        merged_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

def get_domain_sequence(args):
    if args.dataset in ['cifar10c', 'cifar100c']:
        return CORRUPTIONS
    elif args.dataset == 'office_home':
        return DOMAINS
    else:
        return ['clean']