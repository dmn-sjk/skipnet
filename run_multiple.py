import subprocess
import yaml
import sys
from typing import List
from datasets.cifar_c_dataset import CORRUPTIONS
from datasets.office_home_dataset import DOMAINS
import os


TMP_CFG_DIR = '/tmp/skipnet_cfgs'
DEF_CONFIG = {    
    'exp_name': 'def',

    'cmd': 'train', # [train, test]
    'arch': 'cifar10_resnet_38', # cifar100_resnet_38
    'gate_type': 'rnn', # [rnn, ff]
    'dataset': 'cifar10c', # [cifar10, cifar100],
    'data_root': '/datasets',
    'workers': 4,
    'iters': 64000,
    'start_iter': 0,
    'batch_size': 128,
    'lr': 0.05,
    'momentum': 0.9,
    'weight_decay': 1.e-4,
    'print_freq': 10,
    'resume': '', # path to  latest checkpoint
    'pretrained': False,
    'step_ratio': 0.1, # ratio for learning rate reduction'
    'warm_up': False,
    'save_folder': 'save_checkpoints',
    'eval_every': 50,
    'verbose': False,
    'seed': 1,

    'severity': 5,
    'domain': 'gaussian_noise',
    }

def run_script_with_args(script_path: str, arguments: List[str]) -> None:
    """
    Runs a Python script sequentially for each argument in the provided list.
    
    Args:
        script_path (str): Path to the Python script to run
        arguments (List[str]): List of arguments to pass to the script one at a time
    """
    for arg in arguments:
        print(f"\nRunning script with argument: {arg}")
        try:
            # Run the script and wait for it to complete
            process = subprocess.Popen(
                [sys.executable, script_path, str(arg)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for the process to complete and get output
            stdout, stderr = process.communicate()
            
            # Check if the process was successful
            if process.returncode == 0:
                if stdout:
                    print("Output:")
                    print(stdout)
            else:
                print(f"Error running script with argument {arg}:")
                print(f"Exit code: {process.returncode}")
                if stderr:
                    print("Error output:")
                    print(stderr)
                
        except Exception as e:
            print(f"Unexpected error with argument {arg}: {str(e)}")

def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

def modify_and_save_config(cfg_path, domain):
    cfg = DEF_CONFIG
    
    to_update = {
        'exp_name': 'def_' + domain,
        'arch': 'office_home_resnet34',
        'dataset': 'office_home',
        'domain': domain
    }

    cfg.update(to_update)
     
    save_config(cfg, cfg_path)

if __name__ == "__main__":
    # Example usage
    script_to_run = "train_base.py"
    
    os.makedirs(TMP_CFG_DIR, exist_ok=True)
    
    configs_list = {domain: os.path.join(TMP_CFG_DIR, f'{domain}.yaml') for domain in DOMAINS}
    for domain, path in configs_list.items():
        modify_and_save_config(path, domain)
    
    run_script_with_args(script_to_run, list(configs_list.values()))