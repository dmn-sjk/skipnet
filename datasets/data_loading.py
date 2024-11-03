from datasets.cifar_dataset import get_cifar_100_dataloader, get_cifar_10_dataloader
from datasets.cifar_c_dataset import get_cifar_10_c_dataloader, get_cifar_100_c_dataloader


def get_dataloader(args, split):
    if args.dataset == 'cifar100':
        return get_cifar_100_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers)
    elif args.dataset == 'cifar10':
        return get_cifar_10_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers)
    elif args.dataset == 'cifar10c':
        return get_cifar_10_c_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers, 
                                        corruption=args.corruption, severity=args.severity)
    elif args.dataset == 'cifar100c':
        return get_cifar_100_c_dataloader(data_root=args.data_root, split=split, bs=args.batch_size, num_workers=args.workers, 
                                          corruption=args.corruption, severity=args.severity)
    else:
        raise NotImplementedError(args.dataset)