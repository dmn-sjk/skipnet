import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datasets.cifar_c_dataset import CORRUPTIONS


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
    im = ax.imshow(torch.stack(tensor_list, dim=0).numpy(), cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set title and axis labels
    ax.set_title('Average skip mask for each domain')
    ax.set_xlabel('Layers')
    ax.set_ylabel('Domains')
    
    plt.tight_layout()
    
    # Save the heatmap as a PNG file
    plt.savefig(output_file, dpi=300)
    print(f"Heatmap saved to: {output_file}")

# Example usage
dir = 'avg_masks'
masks = []
for i in range(len(CORRUPTIONS)):
    masks.append(torch.load(os.path.join(dir, f"{i}_avg_mask.pth")))
create_and_save_heatmap(masks, "heatmap.png")
