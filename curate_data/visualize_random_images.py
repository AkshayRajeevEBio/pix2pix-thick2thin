import os
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
import random
import torch

def visualize_sample_pairs(
    input_dir: Union[str, Path],
    target_dir: Union[str, Path],
    num_samples: int = 30,
    save_path: Union[str, Path, None] = None
) -> None:
    """
    Visualizes a grid of randomly selected image-target pairs.

    Args:
        input_dir: Directory with input images.
        target_dir: Directory with target images.
        num_samples: Number of image pairs to visualize.
        save_path: Path to save the visualization grid (optional).
    """
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)

    filenames = sorted(os.listdir(input_dir))
    sample_files = random.sample(filenames, min(num_samples, len(filenames)))

    input_images = []
    target_images = []

    for filename in sample_files:
        input_img = Image.open(input_dir / filename).convert("RGB")
        target_img = Image.open(target_dir / filename).convert("RGB")
        input_images.append(input_img)
        target_images.append(target_img)

    cols = 4
    rows = (len(sample_files) + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Random Input-Target Pairs", fontsize=16)

    for idx in range(rows * 2):
        row = idx // 2
        col = (idx % 2) * 2

        if idx < len(sample_files):
            ax_input = axes[row, col] if rows > 1 else axes[col]
            ax_target = axes[row, col + 1] if rows > 1 else axes[col + 1]

            ax_input.imshow(input_images[idx])
            ax_input.set_title("Input")
            ax_input.axis("off")

            ax_target.imshow(target_images[idx])
            ax_target.set_title("Target")
            ax_target.axis("off")
        else:
            for c in range(col, col + 2):
                ax = axes[row, c] if rows > 1 else axes[c]
                ax.axis("off")

    plt.tight_layout(pad=2)
    plt.subplots_adjust(top=0.93)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"✅ Visualization saved to: {save_path}")
    else:
        plt.show()

def main(args):

    visualize_sample_pairs(args.image_dir, args.target_dir,
                               save_path=args.viz_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate data for Pix2Pix model.")
    parser.add_argument("-i", dest="image_dir", type=str,
        default='/inst/rajeev/pixtopix/dataset_unwrapped_90_10/train/input')
    parser.add_argument("-t", dest="target_dir", type=str,
        default='/inst/rajeev/pixtopix/dataset_unwrapped_90_10/train/target')
    parser.add_argument(
        "-v", "--viz-out", dest="viz_out", type=str,
        help="Path to save a visualization grid of training data"
    )
    

    args = parser.parse_args()
    main(args)
