import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import wandb
from cleanfid import fid
import tempfile
from torchvision.utils import save_image
import os
from typing import Optional


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down: bool = True,
        use_dropout: bool = False,
        activation: str = 'relu'
    ) -> None:
        super().__init__()
        if down:
            layers = [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            ]
        # else:
        #     layers = [
        #         nn.ConvTranspose2d(
        #             in_channels, out_channels, 4, 2, 1, bias=False
        #         ),
        #         nn.BatchNorm2d(out_channels)
        #     ]
        else:
            layers = [
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(in_channels, out_channels, 3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ]

        if activation == 'relu':
            layers.insert(0, nn.ReLU(inplace=False))
        elif activation == 'leaky':
            layers.insert(0, nn.LeakyReLU(0.2, inplace=False))

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64
    ) -> None:
        super().__init__()
        self.down1 = UNetBlock(in_channels, features, True, activation='leaky')
        self.down2 = UNetBlock(features, features * 2, True, activation='leaky')
        self.down3 = UNetBlock(features * 2, features * 4, True, activation='leaky')
        self.down4 = UNetBlock(features * 4, features * 8, True, activation='leaky')
        self.down5 = UNetBlock(features * 8, features * 8, True, activation='leaky')
        self.down6 = UNetBlock(features * 8, features * 8, True, activation='leaky')
        self.down7 = UNetBlock(features * 8, features * 8, True, activation='leaky')
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(inplace=False)
        )
        self.up1 = UNetBlock(features * 8, features * 8, False, use_dropout=True)
        self.up2 = UNetBlock(features * 16, features * 8, False, use_dropout=True)
        self.up3 = UNetBlock(features * 16, features * 8, False, use_dropout=True)
        self.up4 = UNetBlock(features * 16, features * 8, False)
        self.up5 = UNetBlock(features * 16, features * 4, False)
        self.up6 = UNetBlock(features * 8, features * 2, False)
        self.up7 = UNetBlock(features * 4, features, False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        return self.final(torch.cat([up7, d1], dim=1))


class ShallowUNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64
    ) -> None:
        super().__init__()
        self.down1 = UNetBlock(in_channels, features, True, activation='leaky')
        self.down2 = UNetBlock(features, features * 2, True, activation='leaky')
        self.down3 = UNetBlock(features * 2, features * 4, True, activation='leaky')

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 4, features * 4, 4, 2, 1),
            nn.ReLU(inplace=False)
        )

        # Dropout enabled on first 2 decoder blocks
        self.up1 = UNetBlock(
            features * 4, features * 4, down=False, use_dropout=True
        )
        self.up2 = UNetBlock(
            features * 8, features * 2, down=False, use_dropout=True
        )
        self.up3 = UNetBlock(
            features * 4, features, down=False, use_dropout=False
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def center_crop(
        self, enc_feat: torch.Tensor, target_feat: torch.Tensor
    ) -> torch.Tensor:
        _, _, h, w = target_feat.shape
        return enc_feat[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        bottleneck = self.bottleneck(d3)

        up1 = self.up1(bottleneck)
        d3 = self.center_crop(d3, up1)
        up2 = self.up2(torch.cat([up1, d3], dim=1))

        d2 = self.center_crop(d2, up2)
        up3 = self.up3(torch.cat([up2, d2], dim=1))

        d1 = self.center_crop(d1, up3)
        return self.final(torch.cat([up3, d1], dim=1))



best_val_loss = float("inf")
best_fid = float("inf")

def update_best_val_loss(val_loss: float) -> bool:
    global best_val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        return True
    return False

def update_best_fid(new_fid: float) -> bool:
    global best_fid
    if new_fid < best_fid:
        best_fid = new_fid
        return True
    return False


def log_val_predictions(
    G: nn.Module,
    dataset,
    device: torch.device,
    epoch: int,
    num_samples: int = 20,
    label: str = "G"
) -> None:
    G.eval()
    indices = torch.randperm(len(dataset))[:num_samples]
    samples = [dataset[i] for i in indices]

    triplet_images = []
    for input_img, target_img, _, _ in samples:
        input_tensor = input_img.unsqueeze(0).to(device)
        pred_tensor = G(input_tensor).cpu().squeeze(0)
        _, h, w = pred_tensor.shape
        input_img = input_img[:, :h, :w]
        target_img = target_img[:, :h, :w]
        pred_tensor = pred_tensor[:, :h, :w]

        def to_pil(x: torch.Tensor) -> Image.Image:
            return TF.to_pil_image((x * 0.5 + 0.5).clamp(0, 1))

        input_pil = to_pil(input_img)
        target_pil = to_pil(target_img)
        pred_pil = to_pil(pred_tensor)

        triplet_width = input_pil.width * 3 + 40
        triplet_height = input_pil.height + 30
        triplet = Image.new('RGB', (triplet_width, triplet_height), 'white')
        draw = ImageDraw.Draw(triplet)
        draw.text((10, 5), "Input", fill='black')
        draw.text((input_pil.width + 20, 5), "Target", fill='black')
        draw.text((2 * input_pil.width + 30, 5), "Pred", fill='black')
        triplet.paste(input_pil, (0, 30))
        triplet.paste(target_pil, (input_pil.width + 10, 30))
        triplet.paste(pred_pil, (2 * input_pil.width + 20, 30))
        triplet_images.append(triplet)

    rows = (len(triplet_images) + 3) // 4
    img_width = triplet_images[0].width
    img_height = triplet_images[0].height
    grid_image = Image.new(
        'RGB', (img_width * 4 + 30, img_height * rows + 10 * (rows - 1)), 'white'
    )

    for idx, triplet in enumerate(triplet_images):
        row = idx // 4
        col = idx % 4
        x = col * (img_width + 10)
        y = row * (img_height + 10)
        grid_image.paste(triplet, (x, y))

    wandb.log({
        f"val_predictions/{label}_epoch_{epoch + 1}": wandb.Image(grid_image),
        "epoch": epoch + 1
    })
    G.train()


def log_val_predictions_comparison(
    G: nn.Module,
    G_ema: Optional[nn.Module],
    dataset,
    device: torch.device,
    epoch: int,
    num_samples: int = 20,
) -> None:
    G.eval()
    if G_ema is not None:
        G_ema.eval()

    indices = torch.randperm(len(dataset))[:num_samples]
    samples = [dataset[i] for i in indices]

    triplet_images = []
    for input_img, target_img, _, _ in samples:
        input_tensor = input_img.unsqueeze(0).to(device)
        input_img = input_img.cpu()
        target_img = target_img.cpu()

        with torch.no_grad():
            pred_G = G(input_tensor).cpu().squeeze(0)
            pred_Gema = G_ema(input_tensor).cpu().squeeze(0) if G_ema is not None else None

        def to_pil(x): return TF.to_pil_image((x * 0.5 + 0.5).clamp(0, 1))
        imgs = [to_pil(img) for img in [input_img, target_img, pred_G]]
        if pred_Gema is not None:
            imgs.append(to_pil(pred_Gema))

        labels = ["Input", "Target", "G", "G_ema"] if pred_Gema is not None else ["Input", "Target", "Pred"]
        total_width = imgs[0].width * len(imgs) + 10 * (len(imgs) - 1)
        total_height = imgs[0].height + 30

        canvas = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(canvas)

        for i, (label, img) in enumerate(zip(labels, imgs)):
            draw.text((i * (img.width + 10), 5), label, fill="black")
            canvas.paste(img, (i * (img.width + 10), 30))

        triplet_images.append(canvas)

    # Create a grid of side-by-side triplets
    rows = (len(triplet_images) + 3) // 4
    img_w, img_h = triplet_images[0].size
    grid = Image.new("RGB", ((img_w + 10) * 4 - 10, (img_h + 10) * rows - 10), "white")
    for idx, img in enumerate(triplet_images):
        x = (idx % 4) * (img_w + 10)
        y = (idx // 4) * (img_h + 10)
        grid.paste(img, (x, y))

    wandb.log({f"val_predictions/epoch_{epoch}": wandb.Image(grid)}, commit=False)



def compute_fid(
    G: nn.Module,
    dataset,
    device: torch.device,
    num_samples: int = 100
) -> float:
    G.eval()
    indices = torch.randperm(len(dataset))[:num_samples]
    samples = [dataset[i] for i in indices]

    with tempfile.TemporaryDirectory() as gen_dir, tempfile.TemporaryDirectory() as real_dir:
        for i, (input_img, target_img, _, _) in enumerate(samples):
            input_tensor = input_img.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_tensor = G(input_tensor).cpu().squeeze(0)

            _, h, w = pred_tensor.shape
            pred_tensor = pred_tensor[:, :h, :w]
            target_img = target_img[:, :h, :w]

            save_image((pred_tensor * 0.5 + 0.5).clamp(0, 1),
                       os.path.join(gen_dir, f"{i:05}.png"))
            save_image((target_img * 0.5 + 0.5).clamp(0, 1),
                       os.path.join(real_dir, f"{i:05}.png"))

        return fid.compute_fid(gen_dir, real_dir)
