import os
import argparse
from typing import List, Tuple
from collections import Counter

import torch
from torch import nn, Tensor
from torchvision import transforms
from torchvision.utils import make_grid
import wandb

from models.generator import ShallowUNetGenerator, compute_fid
from datasets.combined_image_dataset import CombinedImageDataset


def compute_classwise_l1(
    G: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k: int = 6
) -> None:
    G.eval()
    criterion = nn.L1Loss(reduction="none")

    total_loss = 0.0
    class_loss = {"healthy": 0.0, "unhealthy": 0.0}
    class_counts = {"healthy": 0, "unhealthy": 0}
    losses_per_class = {"healthy": [], "unhealthy": []}
    samples_per_class = {"healthy": [], "unhealthy": []}

    with torch.no_grad():
        for input_img, target_img, label, filename in dataloader:
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            pred = G(input_img)
            _, _, h, w = pred.shape
            pred = pred[:, :, :h, :w]
            target_img = target_img[:, :, :h, :w]

            loss_map = criterion(pred, target_img)
            batch_loss = loss_map.mean(dim=[1, 2, 3]).item()

            total_loss += batch_loss
            class_loss[label[0]] += batch_loss
            class_counts[label[0]] += 1

            losses_per_class[label[0]].append(batch_loss)
            samples_per_class[label[0]].append((input_img.cpu(), target_img.cpu(), pred.cpu()))

    print("\n🧪 Test Results:")
    print(f"  Avg L1 loss (all): {total_loss / len(dataloader):.4f}")
    for cls in ["healthy", "unhealthy"]:
        avg = class_loss[cls] / max(1, class_counts[cls])
        print(f"  Avg L1 loss ({cls}): {avg:.4f}")

        # Visualize worst samples
        losses = losses_per_class[cls]
        samples = samples_per_class[cls]
        if not losses:
            continue

        top_indices = sorted(range(len(losses)),
                             key=lambda i: losses[i],
                             reverse=True)[:top_k]
        triplets = [samples[i] for i in top_indices]

        grid = []
        for input_img, target_img, pred in triplets:
            row = torch.cat([
                input_img[0], target_img[0], pred[0]
            ], dim=2)
            grid.append(row)

        grid_img = make_grid(grid, nrow=1, normalize=True, scale_each=True)
        wandb.log({
            f"test/worst_{cls}_samples": wandb.Image(
                grid_img, caption=f"Worst {cls} samples"
            )
        })


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.CenterCrop((75, 75)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = CombinedImageDataset(
        root_dir=os.path.join(args.data_dir, "test"),
        transform=transform,
        label_json=os.path.join(args.data_dir, "labels_test.json")
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    G = ShallowUNetGenerator()
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G = G.to(device)

    if args.use_wandb:
        wandb.init(project="rajeev_p2p", name="test_evaluation")

    compute_classwise_l1(G, loader, device, top_k=args.top_k)

    fid_score = compute_fid(G, dataset, device)
    print(f"\n📏 Test FID: {fid_score:.2f}")

    if args.use_wandb:
        wandb.log({"test/fid": fid_score})
        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset root with test/ and labels_test.json")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained generator .pth file")
    parser.add_argument("--top-k", type=int, default=6,
                        help="Number of worst samples to visualize per class")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    main(args)
