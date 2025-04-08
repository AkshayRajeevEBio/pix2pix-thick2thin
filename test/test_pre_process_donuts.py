import os
import sys
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pix_to_pix.curate_data.pre_process_donuts import CenterAndPad

def visualize_input_target_centered_grid(
    input_dir,
    target_dir,
    transform,
    num_samples=5,
    save_path="centered_debug_grid.png"
):
    """
    Visualize input and target images before and after CenterAndPad.

    Args:
        input_dir (str or Path): Directory with input images.
        target_dir (str or Path): Directory with target images.
        transform (callable): CenterAndPad transform.
        num_samples (int): Number of image pairs to visualize.
        save_path (str): Output path to save the visualization image.
    """
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)

    input_files = sorted(input_dir.glob("*.jpg"))
    target_files = sorted(target_dir.glob("*.jpg"))

    assert len(input_files) == len(target_files), "Mismatched input/target pairs"

    filenames = [f.name for f in input_files]
    selected_filenames = random.sample(filenames, min(num_samples, len(filenames)))

    fig, axes = plt.subplots(
        num_samples, 4, figsize=(4 * 4, num_samples * 4)
    )
    fig.suptitle("Input / Target → Centered Input / Centered Target", fontsize=16)

    for i, filename in enumerate(selected_filenames):
        input_img = Image.open(input_dir / filename).convert("RGB")
        target_img = Image.open(target_dir / filename).convert("RGB")

        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        input_centered = transform(input_tensor)
        target_centered = transform(target_tensor)

        imgs = [
            TF.to_pil_image(input_tensor),
            TF.to_pil_image(target_tensor),
            TF.to_pil_image(input_centered),
            TF.to_pil_image(target_centered)
        ]

        titles = ["Input", "Target", "Centered Input", "Centered Target"]

        for j in range(4):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            ax.imshow(imgs[j])
            ax.set_title(titles[j])
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved grid visualization to: {save_path}")

transform = CenterAndPad(
    lumen_index=(255, 0, 0),
    size=(64, 64),
    constant_values=0
)

visualize_input_target_centered_grid(
    input_dir="/inst/rajeev/pix2pix/datasets_donut_90_10/train/input",
    target_dir="/inst/rajeev/pix2pix/datasets_donut_90_10/train/target",
    transform=transform,
    num_samples=5,
    save_path="/inst/rajeev/pix2pix/test_viz/centered_debug_grid.png"
)
