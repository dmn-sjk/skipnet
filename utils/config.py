import os
import yaml
from types import SimpleNamespace
import argparse


def get_config():
    # TODO: add possibility for command line args overwriting the values from .yaml config
    args = parse_args()
    return load_config(args.cfg_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training with gating')
    parser.add_argument('cfg_path', type=str)
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from a YAML file and convert it to an args-like object.
    
    Args:
        config_path (str): Path to the YAML configuration file
    
    Returns:
        SimpleNamespace: An object with configuration parameters as attributes
    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Read the YAML file
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            raise
    
    # Convert the config dictionary to a SimpleNamespace
    args = dict_to_namespace(config)
    
    return args

def save_config(cfg: SimpleNamespace, save_path):
    """
    Save SimpleNamespace config object to YAML file
    
    Args:
        config: SimpleNamespace object containing configuration
        yaml_path: Path where to save the YAML file
    """
    
    try:
        config_dict = namespace_to_dict(cfg)
    except Exception as e:
        raise ValueError(f"Failed to convert config to dict: {e}")
        
    try:
        with open(save_path, 'w') as file:
            yaml.safe_dump(config_dict, file)
    except Exception as e:
        raise IOError(f"Failed to save YAML file: {e}")


def dict_to_namespace(d):
    """
    Convert nested dictionaries to nested SimpleNamespaces
    """

    if not isinstance(d, dict):
        return d
    
    # Recursively convert nested dictionaries
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def namespace_to_dict(namespace):
    """
    Recursively convert SimpleNamespace to dictionary
    """
    if not isinstance(namespace, SimpleNamespace):
        return namespace
    
    result = {}
    for key, value in vars(namespace).items():
        if isinstance(value, SimpleNamespace):
            result[key] = namespace_to_dict(value)
        elif isinstance(value, list):
            result[key] = [namespace_to_dict(item) if isinstance(item, SimpleNamespace) else item for item in value]
        else:
            result[key] = value
    return result
    

if __name__ == "__main__":
    print(load_config('configs/train_rl.yaml'))
    
    print(__dict__)