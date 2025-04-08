"""Dataset preparation utility with train/val/test split and centering."""

import sys
import os
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Union, Optional, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import torch

from pre_process_donuts import CenterAndPad

DEFAULT_SAVE_DIR = '/inst/rajeev/pix2pix'


def save_visualization_grid(combined_dir: Path,
                            save_path: Path,
                            labels_path: Path,
                            num_grids: int = 3,
                            samples_per_grid: int = 20) -> None:
    with open(labels_path, 'r') as f:
        label_map = json.load(f)

    # Group filenames by label
    label_to_files = {"healthy": [], "unhealthy": []}
    for fname, label in label_map.items():
        if label in label_to_files:
            label_to_files[label].append(combined_dir / fname)

    for label, files in label_to_files.items():
        random.shuffle(files)
        grid_sets = [
            files[i * samples_per_grid:(i + 1) * samples_per_grid]
            for i in range(min(num_grids, len(files) // samples_per_grid))
        ]

        for i, sample_files in enumerate(grid_sets):
            sample_imgs = [Image.open(f).convert("RGB") for f in sample_files]

            widths, heights = zip(*(img.size for img in sample_imgs))
            max_width = max(widths)
            max_height = max(heights)

            cols = 4
            rows = (len(sample_imgs) + cols - 1) // cols
            padding = 10

            grid_width = cols * max_width + (cols - 1) * padding
            grid_height = rows * max_height + (rows - 1) * padding

            grid_img = Image.new("RGB", (grid_width, grid_height),
                                 color=(255, 255, 255))

            for idx, img in enumerate(sample_imgs):
                row = idx // cols
                col = idx % cols
                x = col * (max_width + padding)
                y = row * (max_height + padding)
                grid_img.paste(img, (x, y))

            grid_path = save_path / f"combined_sample_grid_{label}_{i + 1:02}.png"
            grid_img.save(grid_path)
            print(f"✅ Saved {label} visualization grid to: {grid_path}")



class DatasetPreparer:
    def __init__(self,
                 json_path: Union[str, Path],
                 output_dir: Union[str, Path] = "mydataset",
                 split_ratio: Tuple[int, int, int] = (80, 10, 10),
                 seed: int = 42,
                 make_donut: bool = False,
                 center_transform: Optional[Callable] = None) -> None:
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio
        self.seed = seed
        self.make_donut = make_donut
        self.center_transform = center_transform

        with open(self.json_path, 'r', encoding='utf-8') as file:
            raw_map = json.load(file)
            self.data: List[Tuple[str, str, str]] = [
                (k, v, "healthy") for k, v in raw_map["healthy"].items()
            ] + [
                (k, v, "unhealthy") for k, v in raw_map["unhealthy"].items()
            ]

    def _process_path(self, path_str: str) -> str:
        return path_str.replace("unwrapped", "donut") if self.make_donut else path_str

    def _prepare_folders(self) -> None:
        for phase in ('train', 'val', 'test'):
            path = self.output_dir / phase
            path.mkdir(parents=True, exist_ok=True)

    def _apply_transform(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img_tensor = TF.to_tensor(img)
        if self.center_transform:
            img_tensor = self.center_transform(img_tensor)
        return img_tensor

    def _copy_and_transform(
        self,
        dataset: List[Tuple[str, str, str]],
        phase: str
    ) -> None:
        print(f"📂 Saving combined images for {phase}...")
        labels_map = {}
        for index, (inp, tgt, label) in enumerate(
            tqdm(dataset, desc=f"Processing {phase}", unit="file")
        ):
            inp_path = self._process_path(inp)
            tgt_path = self._process_path(tgt)
            try:
                input_tensor = self._apply_transform(inp_path)
                target_tensor = self._apply_transform(tgt_path)
            except Exception as e:
                print(f"⚠ Skipped {inp_path} or {tgt_path}: {e}")
                continue

            filename = f"{index + 1:05}.png"
            input_img = TF.to_pil_image(input_tensor)
            target_img = TF.to_pil_image(target_tensor)

            combined_width = input_img.width + target_img.width
            max_height = max(input_img.height, target_img.height)

            combined_img = Image.new("RGB", (combined_width, max_height))
            combined_img.paste(input_img, (0, 0))
            combined_img.paste(target_img, (input_img.width, 0))

            combined_img.save(self.output_dir / phase / filename)
            labels_map[filename] = label

        # Save labels as a JSON file
        label_path = self.output_dir / f"labels_{phase}.json"
        with open(label_path, "w") as f:
            json.dump(labels_map, f, indent=2)
        print(f"✅ Saved label mapping to {label_path}")


    def _split_data(self) -> Tuple[List, List, List]:
        random.seed(self.seed)
        random.shuffle(self.data)
        total = len(self.data)
        train_pct, val_pct, test_pct = self.split_ratio
        assert train_pct + val_pct + test_pct == 100, (
            "Split ratio must sum to 100"
        )
        train_end = int((train_pct / 100.0) * total)
        val_end = train_end + int((val_pct / 100.0) * total)
        return (
            self.data[:train_end],
            self.data[train_end:val_end],
            self.data[val_end:]
        )

    def run(self) -> None:
        train_data, val_data, test_data = self._split_data()
        self._prepare_folders()
        print(f"📦 Train: {len(train_data)}")
        self._copy_and_transform(train_data, "train")
        print(f"📦 Val: {len(val_data)}")
        self._copy_and_transform(val_data, "val")
        print(f"📦 Test: {len(test_data)}")
        self._copy_and_transform(test_data, "test")

def main(args):
    output_path = Path(DEFAULT_SAVE_DIR) / args.out_dir_name
    if output_path.exists():
        response = input(
            f"Output directory '{output_path}' exists. Overwrite? [y/n]: ")
        if response.strip().lower() != "y":
            print("✖ Choose a different output folder name and try again.")
            return
        print("Removing existing tree...")
        try:
            shutil.rmtree(output_path)
        except Exception as exc:
            print(f"⚠ Failed to remove '{output_path}': {exc}")
            return

    center_transform = CenterAndPad(lumen_index=(255, 0, 0), size=(75, 75))
    prep = DatasetPreparer(
        json_path=args.image_map,
        output_dir=output_path,
        split_ratio=(80, 10, 10),
        make_donut=args.make_donut,
        center_transform=center_transform
    )
    prep.run()

    viz_dir = output_path / "random_samples_visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for phase in ["train", "val", "test"]:
        save_visualization_grid(
            combined_dir=output_path / phase,
            save_path=viz_dir,
            labels_path=output_path / f"labels_{phase}.json",
            num_grids=3,
            samples_per_grid=20
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate Pix2Pix data with centering.")
    parser.add_argument(
        "-i", "--image-map", dest="image_map", type=str,
        required=True, help="Path to JSON with input-target pairs.")
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir_name", type=str,
        required=True, help="Output folder name.")
    parser.add_argument(
        "--make-donut", action="store_true",
        help="Replace 'unwrapped' with 'donut' in image paths.")
    args = parser.parse_args()
    main(args)
