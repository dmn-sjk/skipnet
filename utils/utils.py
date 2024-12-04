import os
import torch
import numpy as np
import random


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', name_prefix=''):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    if is_best:
        save_path = os.path.join(os.path.dirname(filename), name_prefix + '_' * (len(name_prefix) > 0) + 'model_best.pth.tar')
        torch.save(state, save_path)
    else:
        if len(name_prefix) > 0:
            raise NotADirectoryError()
        save_path = filename
        torch.save(state, save_path)
    return save_path
        
def get_save_path(args):
    return os.path.join(args.save_folder, args.arch, args.dataset, args.exp_name)

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)
    
    # Set PyTorch random seed for all GPUs
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic flag for CUDA convolution operations
    torch.backends.cudnn.deterministic = True
    
    # Disable CUDA benchmarking for more deterministic results
    # Note: This might slightly impact performance
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for HashSeed
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_seed_strict(seed):
    set_seed(seed)
    
    # Force PyTorch to use deterministic algorithms
    # Note: This might significantly impact performance
    torch.use_deterministic_algorithms(True)
    
    # Set environment variable for deterministic algorithms
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# If using DataLoader, also set its worker seed
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
