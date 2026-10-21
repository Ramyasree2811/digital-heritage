"""
Unified Training Pipeline for Digital Heritage Restoration
Trains all 5 architectures (U-Net, DnCNN, Pix2Pix, Restormer, SwinIR)
under standardized, reproducible experimental conditions with EarlyStopping,
ReduceLROnPlateau, AdamW, and L1 + SSIM restoration loss.
"""

import os
import sys

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

import argparse
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.unet.model import UNet
from models.dncnn.model import DnCNN
from models.pix2pix.model import Pix2PixGenerator, PatchGANDiscriminator
from models.restormer.model import Restormer
from models.swinir.model import SwinIR

from training.utils import (
    HeritageDataset, get_device, print_device_info, set_seed,
    compute_psnr, save_reproducibility_info
)
from evaluation.ssim import compute_ssim

RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
TRAIN_HIST_DIR = os.path.join(RESULTS_DIR, "training_history")
CHECKPOINTS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "checkpoints")
os.makedirs(TRAIN_HIST_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# =========================================================================
# Image Restoration Loss for Historical Documents
# L1 Loss (Pixel Accuracy) + 0.20 * (1 - SSIM) (Structural Stroke Fidelity)
# =========================================================================
class HeritageDocumentRestorationLoss(nn.Module):
    """
    Combined loss for heritage document restoration:
    - L1 Loss: penalizes absolute pixel difference, avoiding over-smoothing of ink strokes.
    - SSIM Loss (weight = 0.20): penalizes structural distortion, preserving character
      stroke geometry, vowel diacritics, and paper texture.
    """
    def __init__(self, ssim_weight=0.20):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        
        # Approximate differentiable SSIM via local pooling
        pred_mean = F.avg_pool2d(pred, kernel_size=11, stride=1, padding=5)
        target_mean = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
        
        pred_var = F.avg_pool2d(pred * pred, kernel_size=11, stride=1, padding=5) - pred_mean * pred_mean
        target_var = F.avg_pool2d(target * target, kernel_size=11, stride=1, padding=5) - target_mean * target_mean
        covar = F.avg_pool2d(pred * target, kernel_size=11, stride=1, padding=5) - pred_mean * target_mean
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * pred_mean * target_mean + c1) * (2 * covar + c2)) /                    ((pred_mean ** 2 + target_mean ** 2 + c1) * (pred_var + target_var + c2) + 1e-6)
        ssim_loss = 1.0 - ssim_map.mean()
        
        total_loss = l1 + self.ssim_weight * ssim_loss
        return total_loss, l1.item(), ssim_loss.item()

def get_model(model_name):
    clean = model_name.lower().replace("-", "").replace("_", "")
    if clean == "unet":
        return UNet(in_channels=3, out_channels=3), "U-Net"
    elif clean == "dncnn":
        return DnCNN(in_channels=3, out_channels=3, num_layers=8, num_features=32), "DnCNN"
    elif clean in ["pix2pix", "pix2pixgan"]:
        return Pix2PixGenerator(in_channels=3, out_channels=3), "Pix2Pix"
    elif clean == "restormer":
        return Restormer(in_channels=3, out_channels=3, num_blocks=[1, 1, 1], channels=32), "Restormer"
    elif clean == "swinir":
        return SwinIR(in_channels=3, out_channels=3, img_size=256, channels=32, num_heads=4), "SwinIR"
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from unet, dncnn, pix2pix, restormer, swinir.")

def train_single_model(model_name, max_epochs=150, batch_size=8, initial_lr=1e-4,
                       early_stopping_patience=15, lr_reduce_patience=5,
                       max_train_samples=None, force=False):
    device = get_device()
    model, canonical_name = get_model(model_name)
    model = model.to(device)
    
    ckpt_sub = os.path.join(CHECKPOINTS_DIR, model_name.lower())
    os.makedirs(ckpt_sub, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_sub, "best.pth")
    results_ckpt_path = os.path.join(TRAIN_HIST_DIR, f"{model_name.lower()}_best.pth")
    
    if os.path.exists(best_ckpt_path) and not force:
        print(f"[*] Checkpoint already exists for {canonical_name} at: {best_ckpt_path}")
        print("    (Use --force to retrain from scratch)")
        return best_ckpt_path

    print("\n" + "=" * 75)
    print(f" TRAINING MODEL: {canonical_name}")
    print(f" Max Epochs: {max_epochs} | LR: {initial_lr} | Batch Size: {batch_size}")
    print(f" EarlyStopping Patience: {early_stopping_patience} | ReduceLROnPlateau Patience: {lr_reduce_patience}")
    print("=" * 75)

    set_seed(42)
    save_reproducibility_info(42)

    # 1. Datasets with paired patch training
    train_ds = HeritageDataset(split="train", max_samples=max_train_samples, patch_size=256)
    val_ds = HeritageDataset(split="validation", max_samples=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=lr_reduce_patience, verbose=True
    )
    criterion = HeritageDocumentRestorationLoss(ssim_weight=0.20)

    # Pix2Pix Discriminator setup if applicable
    is_gan = (canonical_name == "Pix2Pix")
    if is_gan:
        net_d = PatchGANDiscriminator(in_channels=6).to(device)
        optimizer_d = torch.optim.AdamW(net_d.parameters(), lr=initial_lr, weight_decay=1e-5)
        gan_criterion = nn.BCEWithLogitsLoss()

    best_val_psnr = -1.0
    best_epoch = 0
    patience_counter = 0
    history_records = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_l1_accum = 0.0
        train_ssim_loss_accum = 0.0
        batches = 0

        for batch in train_loader:
            clean = batch["clean"].to(device)
            degraded = batch["degraded"].to(device)

            if is_gan:
                # Train Discriminator
                optimizer_d.zero_grad()
                with torch.no_grad():
                    fake = model(degraded)
                real_pair = torch.cat([degraded, clean], dim=1)
                fake_pair = torch.cat([degraded, fake.detach()], dim=1)
                loss_d_real = gan_criterion(net_d(real_pair), torch.ones_like(net_d(real_pair)))
                loss_d_fake = gan_criterion(net_d(fake_pair), torch.zeros_like(net_d(fake_pair)))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                loss_d.backward()
                optimizer_d.step()

                # Train Generator
                optimizer.zero_grad()
                fake = model(degraded)
                fake_pair = torch.cat([degraded, fake], dim=1)
                loss_g_adv = gan_criterion(net_d(fake_pair), torch.ones_like(net_d(fake_pair)))
                recon_loss, l1_val, ssim_val = criterion(fake, clean)
                total_g_loss = recon_loss + 0.05 * loss_g_adv
                total_g_loss.backward()
                optimizer.step()

                train_loss_accum += total_g_loss.item()
                train_l1_accum += l1_val
                train_ssim_loss_accum += ssim_val
            else:
                optimizer.zero_grad()
                out = model(degraded)
                loss, l1_val, ssim_val = criterion(out, clean)
                loss.backward()
                optimizer.step()

                train_loss_accum += loss.item()
                train_l1_accum += l1_val
                train_ssim_loss_accum += ssim_val

            batches += 1

        avg_train_loss = train_loss_accum / max(1, batches)
        avg_train_l1 = train_l1_accum / max(1, batches)

        # Validation phase
        model.eval()
        val_loss_accum = 0.0
        val_psnr_accum = 0.0
        val_ssim_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                clean = batch["clean"].to(device)
                degraded = batch["degraded"].to(device)

                out = model(degraded)
                v_loss, _, _ = criterion(out, clean)
                val_loss_accum += v_loss.item()

                out_np = out.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                clean_np = clean.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

                for b in range(out_np.shape[0]):
                    val_psnr_accum += compute_psnr(clean_np[b], out_np[b])
                    val_ssim_accum += compute_ssim(clean_np[b], out_np[b], data_range=1.0)
                    val_batches += 1

        avg_val_loss = val_loss_accum / max(1, len(val_loader))
        avg_val_psnr = val_psnr_accum / max(1, val_batches)
        avg_val_ssim = val_ssim_accum / max(1, val_batches)

        # Update learning rate scheduler based on validation PSNR
        scheduler.step(avg_val_psnr)
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch
        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "train_l1": round(avg_train_l1, 5),
            "val_loss": round(avg_val_loss, 5),
            "val_psnr": round(avg_val_psnr, 2),
            "val_ssim": round(avg_val_ssim, 4),
            "lr": current_lr
        }
        history_records.append(record)

        print(f"Epoch [{epoch:03d}/{max_epochs:03d}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val PSNR: {avg_val_psnr:.2f} dB | "
              f"Val SSIM: {avg_val_ssim:.4f} | LR: {current_lr:.2e}")

        # Checkpoint & Early Stopping Check
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            ckpt_data = {
                "epoch": epoch,
                "model_name": canonical_name,
                "state_dict": model.state_dict(),
                "val_psnr": avg_val_psnr,
                "val_ssim": avg_val_ssim,
                "optimizer": optimizer.state_dict()
            }
            torch.save(ckpt_data, best_ckpt_path)
            torch.save(ckpt_data, results_ckpt_path)
            print(f"  --> [*] New best validation PSNR: {best_val_psnr:.2f} dB (Saved best checkpoint)")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation PSNR did not improve for {early_stopping_patience} consecutive epochs.")
                print(f"Restoring best checkpoint from epoch {best_epoch} (PSNR: {best_val_psnr:.2f} dB).")
                break

    total_training_time = time.time() - start_time
    print(f"\nTraining for {canonical_name} completed in {total_training_time / 60:.2f} minutes.")
    print(f"Best Checkpoint: Epoch {best_epoch} | Best Val PSNR: {best_val_psnr:.2f} dB")

    # Save training history CSV
    df_hist = pd.DataFrame(history_records)
    csv_hist_path = os.path.join(TRAIN_HIST_DIR, f"{model_name.lower()}_history.csv")
    df_hist.to_csv(csv_hist_path, index=False)
    print(f"Saved training history to: {csv_hist_path}")

    # Plot loss and PSNR curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", color="#1d4ed8")
    plt.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", color="#dc2626")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{canonical_name} Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df_hist["epoch"], df_hist["val_psnr"], label="Val PSNR (dB)", color="#15803d")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title(f"{canonical_name} Validation PSNR")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(TRAIN_HIST_DIR, f"{model_name.lower()}_loss_curve.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved loss curve plot to: {curve_path}")

    return best_ckpt_path

def train_all_models(max_epochs=150, batch_size=8, force=False):
    models = ["unet", "dncnn", "pix2pix", "restormer", "swinir"]
    print("=" * 75)
    print(" DIGITAL HERITAGE RESTORATION — MULTI-MODEL SYSTEM TRAINING")
    print(f" Models to train: {', '.join(models)}")
    print("=" * 75)
    
    for m in models:
        train_single_model(m, max_epochs=max_epochs, batch_size=batch_size, force=force)
        
    print("\n" + "=" * 75)
    print(" ALL 5 RESTORATION ARCHITECTURES HAVE BEEN TRAINED & CHECKPOINTED!")
    print(" Next Step: Run 'python evaluate.py' to compare on the held-out test set.")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Heritage Image Restoration Models")
    parser.add_argument("--model", type=str, default=None, help="Model name (unet, dncnn, pix2pix, restormer, swinir)")
    parser.add_argument("--all", action="store_true", help="Train all 5 restoration models sequentially")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum epochs (default: 150)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=15, help="EarlyStopping patience (default: 15)")
    parser.add_argument("--force", action="store_true", help="Force retrain even if checkpoint exists")
    args = parser.parse_args()

    print_device_info()

    if args.all or (args.model is None):
        train_all_models(max_epochs=args.epochs, batch_size=args.batch_size, force=args.force)
    else:
        train_single_model(args.model, max_epochs=args.epochs, batch_size=args.batch_size,
                           initial_lr=args.lr, early_stopping_patience=args.patience, force=args.force)
