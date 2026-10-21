import os
import sys

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

import argparse
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.pix2pix.model import Pix2PixGenerator, Pix2PixDiscriminator
from training.utils import (
    HeritageDataset, compute_psnr, compute_validation_composite_score,
    print_device_info, set_seed, has_trained_checkpoint,
    save_checkpoint, load_checkpoint, save_training_history_csv,
    plot_training_curves, get_checkpoint_dir
)
from evaluation.ssim import compute_ssim
from evaluation.lpips import compute_lpips

MAX_EPOCHS = 150
PATIENCE = 20
DEFAULT_LR = 2e-4

def train_pix2pix(epochs=MAX_EPOCHS, batch_size=8, lr=DEFAULT_LR, patience=PATIENCE,
                  resume=False, force_retrain=False, max_samples=120):
    model_name = "Pix2Pix"
    ckpt_dir = get_checkpoint_dir(model_name)
    print(f"\n{'='*65}\n  Training Pipeline: {model_name} (Conditional GAN) (Max Epochs: {epochs}, Patience: {patience}, LR: {lr})\n{'='*65}")
    
    # 1. Checkpoint detection guard
    if not force_retrain and not resume and has_trained_checkpoint(model_name):
        print(f"[CHECKPOINT DETECTED] {model_name} already trained at {ckpt_dir}. Skipping training.")
        return True
        
    device = print_device_info()
    set_seed(42)
    
    train_ds = HeritageDataset(split="train", max_samples=max_samples)
    val_ds = HeritageDataset(split="validation", max_samples=30)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    generator = Pix2PixGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = Pix2PixDiscriminator(in_channels=3, out_channels=3).to(device)
    
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixelwise = nn.L1Loss()
    lambda_pixel = 100.0
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    
    start_epoch = 0
    best_score = -1.0
    patience_counter = 0
    history_rows = []
    
    # Resume checkpoint if requested
    gen_latest = os.path.join(ckpt_dir, "latest_generator.pth")
    if resume and os.path.exists(gen_latest):
        print(f"[RESUME] Loading latest Pix2Pix checkpoints from {ckpt_dir}...")
        ckpt_g = load_checkpoint(model_name, is_discriminator=False, device=device, load_latest=True)
        generator.load_state_dict(ckpt_g["state_dict"])
        if ckpt_g.get("optimizer_state"):
            optimizer_G.load_state_dict(ckpt_g["optimizer_state"])
            
        disc_latest = os.path.join(ckpt_dir, "latest_discriminator.pth")
        if os.path.exists(disc_latest):
            ckpt_d = load_checkpoint(model_name, is_discriminator=True, device=device, load_latest=True)
            discriminator.load_state_dict(ckpt_d["state_dict"])
            if ckpt_d.get("optimizer_state"):
                optimizer_D.load_state_dict(ckpt_d["optimizer_state"])
                
        start_epoch = ckpt_g.get("epoch", 0) + 1
        best_score = ckpt_g.get("best_score", -1.0)
        
        hist_csv = os.path.join(ckpt_dir, "training_history.csv")
        if os.path.exists(hist_csv):
            history_rows = pd.read_csv(hist_csv).to_dict("records")
        print(f"[RESUME] Resuming from epoch {start_epoch+1}/{epochs} (Best Score: {best_score:.4f})")
    
    print(f"Starting {model_name} training for epochs {start_epoch+1} to {epochs} on {len(train_ds)} samples...")
    t_start = time.time()
    
    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        g_loss_accum = 0.0
        
        for batch in train_loader:
            deg = batch["degraded"].to(device)
            clean = batch["clean"].to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            fake_clean = generator(deg).detach()
            
            pred_real = discriminator(deg, clean)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            pred_fake = discriminator(deg, fake_clean)
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            gen_clean = generator(deg)
            pred_fake_for_G = discriminator(deg, gen_clean)
            loss_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_pixel = criterion_pixelwise(gen_clean, clean)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            
            loss_G.backward()
            optimizer_G.step()
            g_loss_accum += loss_G.item()
            
        avg_train_loss = g_loss_accum / len(train_loader)
        
        # Validation
        generator.eval()
        val_loss, psnr_list, ssim_list, lpips_list = 0.0, [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                deg = batch["degraded"].to(device)
                clean = batch["clean"].to(device)
                fake = generator(deg)
                val_loss += criterion_pixelwise(fake, clean).item()
                
                fake_c = torch.clamp(fake, 0.0, 1.0)
                psnr_list.append(compute_psnr(fake_c[0], clean[0]).item())
                
                fake_np = (fake_c[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                clean_np = (clean[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                ssim_list.append(compute_ssim(fake_np, clean_np))
                if idx < 5:
                    lpips_list.append(compute_lpips(fake_np, clean_np))
                    
        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))
        avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.45
        composite_score = compute_validation_composite_score(avg_psnr, avg_ssim, avg_lpips)
        cur_lr = optimizer_G.param_groups[0]['lr']
        
        history_rows.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "PSNR": round(avg_psnr, 2),
            "SSIM": round(avg_ssim, 4),
            "LPIPS": round(avg_lpips, 4),
            "composite_score": round(composite_score, 4),
            "learning_rate": cur_lr
        })
        
        print(f"Epoch [{epoch+1}/{epochs}] | G Loss: {avg_train_loss:.4f} | Val PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | Composite: {composite_score:.4f}")
        
        config = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_pixel": lambda_pixel,
            "val_psnr": round(avg_psnr, 2),
            "val_ssim": round(avg_ssim, 4),
            "val_lpips": round(avg_lpips, 4),
            "composite_score": round(composite_score, 4)
        }
        
        # Check for improvement
        if composite_score > (best_score + 1e-4):
            best_score = composite_score
            patience_counter = 0
            save_checkpoint(model_name, generator.state_dict(), optimizer_G.state_dict(), epoch=epoch, best_score=best_score, config=config, is_best=True, is_discriminator=False)
            save_checkpoint(model_name, discriminator.state_dict(), optimizer_D.state_dict(), epoch=epoch, best_score=best_score, config=config, is_best=True, is_discriminator=True)
            print(f"  --> Saved new best checkpoints (Generator + Discriminator): Composite: {best_score:.4f} (PSNR: {avg_psnr:.2f} dB)")
        else:
            patience_counter += 1
            print(f"  --> No improvement for {patience_counter}/{patience} epochs.")
            
        save_checkpoint(model_name, generator.state_dict(), optimizer_G.state_dict(), epoch=epoch, best_score=composite_score, config=config, is_best=False, is_discriminator=False)
        save_checkpoint(model_name, discriminator.state_dict(), optimizer_D.state_dict(), epoch=epoch, best_score=composite_score, config=config, is_best=False, is_discriminator=True)
        save_training_history_csv(model_name, history_rows)
        plot_training_curves(model_name, history_rows)
        
        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] {model_name} validation performance did not improve for {patience} consecutive epochs. Stopping at epoch {epoch+1}.")
            break
            
    print(f"{model_name} training finished in {(time.time()-t_start):.1f}s. Best Composite Score: {best_score:.4f}\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    train_pix2pix(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                  resume=args.resume, force_retrain=args.force_retrain)
