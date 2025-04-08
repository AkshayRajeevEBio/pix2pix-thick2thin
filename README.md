# 🧠 Pix2Pix Artery Reconstruction with Class-Aware Debugging

This repository contains a custom **Pix2Pix** implementation for reconstructing coronary artery cross-sections. It includes a rich training loop with validation, mixed precision, discriminator feature loss, perceptual loss, FID scoring, class-aware debugging, and Weights & Biases logging.

---

## 🚀 Features

- ✅ **Pix2Pix GAN** with U-Net Generator and PatchGAN Discriminator  
- ✅ **EMA** (Exponential Moving Average) tracking for generator  
- ✅ **Validation with metrics:** L1, SSIM, PSNR (class-wise and global)  
- ✅ **Discriminator Feature Matching Loss**  
- ✅ **VGG-based Perceptual Loss**  
- ✅ **Class-aware worst-case sample visualization**  
- ✅ **Side-by-side predictions**: `Input | Target | G | G_ema`  
- ✅ **FID score** via saved images, with early stopping support  
- ✅ **W&B Logging**: losses, gradients, predictions, EMA weights  
- ✅ **Balanced sampling** for handling healthy/unhealthy class imbalance  
- ✅ **GAN Warmup** support

---

## 🗂️ Project Structure

"""
pix_to_pix/
├── train_pix2pix.py               # Main training script
├── models/
│   ├── generator.py               # UNet generator and prediction utilities
│   ├── discriminator.py          # PatchDiscriminator with feature hooks
│   ├── perceptual.py             # VGG feature extractor for perceptual loss
├── datasets/
│   ├── combined_image_dataset.py # Dataset loader for combined input-target images
"""

---

## 🧩 Dataset Format

This model expects **combined input-target image pairs** (side-by-side) with associated class labels.

"""
datasets_donut_80_10_10/
├── train/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── val/
│   ├── image_101.png
│   └── ...
├── test/
│   └── ...
├── labels_train.json             # {"image_001.png": "healthy", ...}
├── labels_val.json
└── labels_test.json              # (optional)

"""

---

### Prepare dataset
Ensure your data follows the format above and is preprocessed into combined images. See DatasetPreparer for help generating this.

### Train a model:
python pix_to_pix/train_pix2pix.py \
  --data-dir /path/to/datasets_donut_80_10_10 \
  --save-dir /path/to/save_checkpoints \
  --batch-size 16 \
  --num-epochs 100 \
  --lambda-perceptual 10.0 \
  --gan-warmup-epochs 10

Enable/Disable Features

Flag	Description
- --no-amp	Disable mixed precision (AMP)
- --no-wandb	Disable Weights & Biases logging
- --no-ema-worst	Use G instead of G_ema for worst-case logging
- --fid-every	How often to compute FID (in epochs)
- --early-stopping-patience	Stop if FID doesn't improve for N rounds

### Metrics & Logging:
Validation loss: val_L1, val_SSIM, val_PSNR
Gradients: grad_norm_G, grad_norm_D
Generator loss: loss_L1, loss_perceptual, loss_G_GAN, loss_feature_matching
Visuals:
- Worst-case reconstructions by class
- Side-by-side comparisons: Input | Target | G | G_ema
- EMA weight drift monitoring

### Dev Notes
- You can edit train_pix2pix.py to switch loss weights or sampling strategies
- Checkpoints are saved every epoch; best FID model is saved separately

Acknowledgements
- Pix2Pix Paper (Isola et al., 2017)
- VGG16 perceptual loss adapted from torchvision
- Feature matching adapted from discriminator feature hooks
