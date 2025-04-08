"""Pix2Pix training loop with validation, W&B logging, AMP support,
FID scoring, and debugging of worst samples."""

import sys
import os
from typing import List, Literal, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import WeightedRandomSampler, DataLoader
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
import numpy as np

from torchvision import transforms
from tqdm import tqdm
import wandb
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.generator import (
    ShallowUNetGenerator as UNetGenerator,
    log_val_predictions,
    log_val_predictions_comparison,
    compute_fid,
    update_best_fid
)
from models.discriminator import PatchDiscriminator
from models.perceptual import VGGFeatureExtractor
from datasets.combined_image_dataset import CombinedImageDataset

GAN_WEIGHT = 0.1

def log_worst_generator_outputs_by_class_buffered(
    samples: List[Tuple[Tensor, Tensor, Tensor, Tensor, str]],
    epoch: int,
    top_k: int = 6
) -> None:
    by_class = {"healthy": [], "unhealthy": []}

    for inp, tgt, fake, loss, label in samples:
        by_class[label].append((loss.item(), inp, tgt, fake))

    for cls, records in by_class.items():
        if not records:
            continue

        top = sorted(records, key=lambda x: x[0], reverse=True)[:top_k]
        grid = []
        for _, inp, tgt, fake in top:
            c = min(inp.shape[0], tgt.shape[0], fake.shape[0])
            h = min(inp.shape[1], tgt.shape[1], fake.shape[1])
            w = min(inp.shape[2], tgt.shape[2], fake.shape[2])
            row = torch.cat([
                inp[:c, :h, :w],
                tgt[:c, :h, :w],
                fake[:c, :h, :w]
            ], dim=2)
            grid.append(row)

        if grid:
            grid_img = torch.cat(grid, dim=1)
            wandb.log({
                f"worst_{cls}_epoch": [
                    wandb.Image(
                        grid_img,
                        caption=f"Worst {cls} samples @ epoch {epoch}"
                    )
                ]
            })

def add_instance_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    if std > 0:
        noise = torch.randn_like(x) * std
        return x + noise
    return x

def update_ema(model_src, model_ema, decay):
    with torch.no_grad():
        for param_src, param_ema in zip(model_src.parameters(), model_ema.parameters()):
            param_ema.data.mul_(decay).add_(param_src.data, alpha=1 - decay)

def safe_amp_step(loss, optimizer, scaler, tag="", global_step=None, verbose=False):
    if loss is None or not loss.requires_grad or not torch.isfinite(loss):
        if verbose:
            print(f"⚠️ Skipping {tag} step at step {global_step} due to invalid loss.")
        return False

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if verbose and tag:
        loss_val = loss.item() if loss.requires_grad else "?"
        scale_val = scaler.get_scale()
        print(f"✅ {tag} step passed: loss={loss_val:.4f}, scale={scale_val:.2e}")
    return True

def log_grad_norm(model, tag, step, use_wandb=True):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if use_wandb:
        wandb.log({f"{tag}_grad_norm": total_norm, "global_step": step})
        
def compute_ssim_psnr(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred = torch.clamp(pred, -1, 1).cpu().numpy()
    target = torch.clamp(target, -1, 1).cpu().numpy()
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    pred = np.transpose(pred.squeeze(), (1, 2, 0))
    target = np.transpose(target.squeeze(), (1, 2, 0))

    ssim = ssim_fn(pred, target, channel_axis=2, data_range=1.0)
    psnr = psnr_fn(pred, target, data_range=1.0)
    return ssim, psnr

def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5



def train_pix2pix(data_dir: str,
                  save_dir: str,
                  device: Literal["cpu", "cuda"] = "cuda",
                  batch_size: int = 16,
                  num_epochs: int = 100,
                  lr: float = 2e-4,
                  lambda_l1: float = 100.0,
                  lambda_perceptual: float = 10.0,
                  use_amp: bool = True,
                  use_wandb: bool = True,
                  fid_every: int = 5,
                  gan_warmup_epochs: int = 0,
                  use_ema_for_worst: bool = True,
                  early_stopping_patience: int = 7) -> None:
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.CenterCrop((75, 75)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_dataset = CombinedImageDataset(
        os.path.join(data_dir, "train"),
        transform=transform,
        label_json=os.path.join(data_dir, "labels_train.json")
    )

    label_counts = Counter(train_dataset.labels.values())
    weight_healthy = 1.0 / label_counts["healthy"]
    weight_unhealthy = 1.0 / label_counts["unhealthy"]
    weights = [
        weight_healthy if label == "healthy" else weight_unhealthy
        for label in train_dataset.labels.values()
    ]

    print(f"📊 Class counts: {label_counts}, "
          f"Sampling weights → healthy: {weight_healthy:.4e}, "
          f"unhealthy: {weight_unhealthy:.4e}")

    val_dataset = CombinedImageDataset(
        os.path.join(data_dir, "val"),
        transform=transform,
        label_json=os.path.join(data_dir, "labels_val.json")
    )

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    G = UNetGenerator().to(device)
    G_ema = UNetGenerator().to(device)
    G_ema.load_state_dict(G.state_dict())  # initialize EMA with same weights
    ema_decay = 0.999  # can be tuned
    D = PatchDiscriminator().to(device)
    vgg_loss = VGGFeatureExtractor().to(device)
    criterion_perceptual = nn.L1Loss()

    torch.backends.cudnn.benchmark = True

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss(reduction='none')

    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))


    scaler_G = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_D = torch.cuda.amp.GradScaler(enabled=use_amp)

    if use_wandb:
        wandb.init(project="rajeev_p2p", config={
            "batch_size": batch_size,
            "epochs": num_epochs,
            "lr": lr,
            "lambda_l1": lambda_l1,
            "lambda_perceptual": lambda_perceptual
        })

    global_step = 0
    initial_std = 0.1
    decay_rate = 0.99
    best_fid = float("inf")
    patience_counter = 0
    for epoch in range(num_epochs):
        torch.cuda.reset_peak_memory_stats()
        G.train()
        D.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        worst_samples_buffer = []

        for input_img, target_img, labels, _ in loop:
            input_img = input_img.to(device, non_blocking=True)
            target_img = target_img.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_ema_for_worst and global_step > 1000:
                    fake_img = G_ema(input_img)
                else:
                    fake_img = G(input_img)
                _, _, h, w = fake_img.shape
                input_img = input_img[:, :, :h, :w]
                target_img = target_img[:, :, :h, :w]
                
                # Add noise to discriminator inputs
                std = initial_std * (decay_rate ** epoch)
                # input_noisy = add_instance_noise(input_img.detach(), std=std)
                # target_noisy = add_instance_noise(target_img.detach(), std=std)
                # fake_noisy = add_instance_noise(fake_img.detach(), std=std)
                input_noisy = input_img.detach()
                target_noisy = target_img.detach()
                fake_noisy = fake_img.detach()

                D_real = D(input_noisy, target_noisy)
                D_fake = D(input_noisy, fake_noisy)

                # D_real = D(input_img, target_img)
                # D_fake = D(input_img, fake_img.detach())

                # smoothed real
                real_labels = torch.empty_like(D_real).uniform_(0.9, 1.0)
                # unchanged fake
                fake_labels = torch.zeros_like(D_fake)

                loss_D_real = criterion_GAN(D_real, real_labels)
                loss_D_fake = criterion_GAN(D_fake, fake_labels)

                loss_D = (loss_D_real + loss_D_fake) / 2

            opt_D.zero_grad()
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            with torch.cuda.amp.autocast(enabled=use_amp):
                D_fake_out, D_fake_feats = D.forward_with_features(input_img, fake_img)
                _, D_real_feats = D.forward_with_features(input_img, target_img)

                # Feature Matching Loss (L1 between intermediate layers)
                fm_loss = sum([
                    nn.functional.l1_loss(f_fake, f_real.detach())
                    for f_fake, f_real in zip(D_fake_feats, D_real_feats)
                ]) * 1.0  # You can tune this multiplier

                loss_G_GAN = criterion_GAN(
                    D_fake_out, torch.ones_like(D_fake_out))

                pixelwise_loss = criterion_L1(
                    fake_img, target_img).mean(dim=[1, 2, 3])
                loss_G_L1 = pixelwise_loss.mean() * lambda_l1

                feat_fake = vgg_loss(fake_img)
                feat_real = vgg_loss(target_img)
                loss_G_perceptual = criterion_perceptual(
                    feat_fake, feat_real) * lambda_perceptual
                # loss_G_perceptual = torch.tensor(0.0, device=device)
                # loss_G = (
                #     (GAN_WEIGHT * loss_G_GAN if epoch >= gan_warmup_epochs else 0)
                #     + lambda_l1 * loss_G_L1
                #     + lambda_perceptual * loss_G_perceptual
                # )
                loss_G = (
                    (GAN_WEIGHT * loss_G_GAN if epoch >= gan_warmup_epochs else 0)
                    + lambda_l1 * loss_G_L1
                    + lambda_perceptual * loss_G_perceptual
                    + fm_loss
                )

            if safe_amp_step(loss_G, opt_G, scaler_G, tag="Generator",
                             global_step=global_step):
                update_ema(G, G_ema, ema_decay)
            else:
                continue

            for i in range(input_img.size(0)):
                worst_samples_buffer.append((
                    input_img[i].detach().cpu(),
                    target_img[i].detach().cpu(),
                    fake_img[i].detach().cpu(),
                    pixelwise_loss[i].detach().cpu(),
                    labels[i]
                ))

            loop.set_postfix({
                "loss_D": loss_D.item(),
                "loss_G": loss_G.item(),
                "loss_L1": loss_G_L1.item()
            })

            if use_wandb and global_step % 10 == 0:
                wandb.log({
                    "grad_norm_G": compute_grad_norm(G),
                    "grad_norm_D": compute_grad_norm(D),
                    "loss_D": loss_D.item(),
                    "loss_G": loss_G.item(),
                    "loss_L1": loss_G_L1.item(),
                    "loss_G_GAN": loss_G_GAN.item(),
                    "loss_perceptual": loss_G_perceptual.item(),
                    "loss_feature_matching": fm_loss.item(),
                    "global_step": global_step
                })
            
            if global_step % 100 == 0:
                for name, param in G_ema.named_parameters():
                    if "weight" in name:
                        wandb.log({f"ema_weights/{name}_mean": param.data.mean().item()})
                        break  # Just one layer is enough to catch issues

            global_step += 1

        G_ema.eval()
        val_loss_total = 0.0
        val_loss_healthy = 0.0
        val_loss_unhealthy = 0.0
        n_healthy = 0
        n_unhealthy = 0
        ssim_total = 0.0
        psnr_total = 0.0

        with torch.no_grad():
            for input_img, target_img, label, _ in val_loader:
                input_img = input_img.to(device, non_blocking=True)
                target_img = target_img.to(device, non_blocking=True)
                fake_img = G_ema(input_img)
                _, _, h, w = fake_img.shape
                fake_img = fake_img[:, :, :h, :w]
                target_img = target_img[:, :, :h, :w]

                l1 = nn.L1Loss(reduction="mean")(fake_img, target_img).item()
                val_loss_total += l1

                if label[0] == "healthy":
                    val_loss_healthy += l1
                    n_healthy += 1
                elif label[0] == "unhealthy":
                    val_loss_unhealthy += l1
                    n_unhealthy += 1
                
                ssim_val, psnr_val = compute_ssim_psnr(fake_img[0], target_img[0])
                ssim_total += ssim_val
                psnr_total += psnr_val

        val_loss_total /= len(val_loader)
        val_loss_healthy /= max(n_healthy, 1)
        val_loss_unhealthy /= max(n_unhealthy, 1)
        avg_ssim = ssim_total / len(val_loader)
        avg_psnr = psnr_total / len(val_loader)

        print(f"📉 Val L1: all={val_loss_total:.4f}, "
              f"healthy={val_loss_healthy:.4f}, "
              f"unhealthy={val_loss_unhealthy:.4f}")

        torch.cuda.empty_cache()

        if use_wandb:
            wandb.log({
                "val_L1": val_loss_total,
                "val_L1_healthy": val_loss_healthy,
                "val_L1_unhealthy": val_loss_unhealthy,
                "val_SSIM": avg_ssim,
                "val_PSNR": avg_psnr,
                "epoch": epoch + 1
            })

        torch.save(G.state_dict(), os.path.join(
            save_dir, f"G_epoch_{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(
            save_dir, f"D_epoch_{epoch+1}.pth"))
        print(f"✅ Saved models for epoch {epoch+1}")

        if use_wandb:
            if use_ema_for_worst:
                log_val_predictions_comparison(G, G_ema, val_dataset, device, epoch)
            else:
                log_val_predictions(G, val_dataset, device, epoch, label="G")

        if (epoch + 1) % fid_every == 0:
            fid_score = compute_fid(G_ema, val_dataset, device)
            print(f"📏 FID score: {fid_score:.2f}")
            if use_wandb:
                wandb.log({"fid": fid_score, "epoch": epoch + 1})
            if update_best_fid(fid_score):
                print("💾 New best FID! 🎯")
                torch.save(G_ema.state_dict(),
                           os.path.join(save_dir, "G_best.pth"))
                log_worst_generator_outputs_by_class_buffered(
                    samples=worst_samples_buffer,
                    epoch=epoch + 1
                )
            if fid_score < best_fid:
                best_fid = fid_score
                patience_counter = 0
                # Save best model checkpoint
                torch.save(G.state_dict(), os.path.join(save_dir, "best_G.pt"))
                torch.save(D.state_dict(), os.path.join(save_dir, "best_D.pt"))
                if G_ema:
                    torch.save(G_ema.state_dict(), os.path.join(save_dir, "best_G_ema.pt"))
                print(f"💾 New best FID: {fid_score:.2f}")
            else:
                patience_counter += 1
                print(f"⚠️ No improvement in FID. Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"⏹️ Early stopping triggered. Best FID: {best_fid:.2f}")
                    break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fid-every", type=int, default=5)
    parser.add_argument("--gan-warmup-epochs", type=int, default=0)
    parser.add_argument("--no-ema-worst", action="store_true")
    parser.add_argument("--lambda-perceptual", type=float, default=10.0)
    parser.add_argument('--early-stopping-patience', type=int, default=7)
    args = parser.parse_args()

    train_pix2pix(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        use_amp=not args.no_amp,
        use_wandb=not args.no_wandb,
        fid_every=args.fid_every,
        gan_warmup_epochs=args.gan_warmup_epochs,
        use_ema_for_worst=not args.no_ema_worst,
        lambda_perceptual=args.lambda_perceptual,
        early_stopping_patience=args.early_stopping_patience,
    )
