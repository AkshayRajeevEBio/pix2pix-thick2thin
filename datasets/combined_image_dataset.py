from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path


class CombinedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_json=None):
        """
        Args:
            root_dir (str or Path): Directory with combined .png images.
            transform (callable, optional): Transform to apply to both halves.
            label_json (str or Path, optional): Path to labels_<phase>.json.
        """
        self.root_dir = Path(root_dir)
        self.image_files = sorted(self.root_dir.glob("*.png"))
        self.transform = transform

        if label_json:
            with open(label_json, 'r') as f:
                self.labels = json.load(f)
        else:
            self.labels = {img.name: "unknown" for img in self.image_files}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img = Image.open(path).convert("RGB")

        w, h = img.size
        input_img = img.crop((0, 0, w // 2, h))
        target_img = img.crop((w // 2, 0, w, h))

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        label = self.labels.get(path.name, "unknown")

        return input_img, target_img, label, path.name
