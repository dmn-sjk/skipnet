import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datasets.cifar_c_dataset import CORRUPTIONS
from datasets.office_home_dataset import DOMAINS


def create_and_save_heatmap(tensor_list, output_file):
    """
    Create a heatmap from a list of 1D tensors and save it as a PNG file.
    
    Parameters:
    tensor_list (list): A list of 1D tensors.
    output_file (str): The path and filename to save the heatmap as a PNG.
    
    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i][:-2]
    im = ax.imshow(torch.stack(tensor_list, dim=0).numpy(), cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set title and axis labels
    # ax.set_title('Average skip mask for each domain\nafter joint training for training split')
    ax.set_title('Average skip mask for each domain\nafter joint training for train split')
    ax.set_xlabel('Layers')
    ax.set_ylabel('Domains')
    
    plt.tight_layout()
    
    # Save the heatmap as a PNG file
    plt.savefig(output_file, dpi=300)
    print(f"Heatmap saved to: {output_file}")

# Example usage
dir = 'avg_masks/office_home/joint_training'
masks = []
# for i in range(len(CORRUPTIONS)):
for i in range(len(DOMAINS)):
    masks.append(torch.load(os.path.join(dir, f"{i}_train_avg_mask.pth")))
create_and_save_heatmap(masks, "joint_trainSplit_heatmap.png")
