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

from models.restormer.model import Restormer
from training.utils import (
    HeritageDataset, HeritageRestorationLoss, compute_psnr,
    compute_validation_composite_score, print_device_info, set_seed,
    has_trained_checkpoint, save_checkpoint, load_checkpoint,
    save_training_history_csv, plot_training_curves, get_checkpoint_dir
)
from evaluation.ssim import compute_ssim
from evaluation.lpips import compute_lpips

MAX_EPOCHS = 75
PATIENCE = 10
DEFAULT_LR = 2e-4

def train_restormer(epochs=MAX_EPOCHS, batch_size=8, lr=DEFAULT_LR, patience=PATIENCE,
                    resume=False, force_retrain=False, max_samples=120):
    model_name = "Restormer"
    ckpt_dir = get_checkpoint_dir(model_name)
    print(f"\n{'='*65}\n  Training Pipeline: {model_name} (Max Epochs: {epochs}, Patience: {patience}, LR: {lr})\n{'='*65}")
    
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
    
    model = Restormer(in_channels=3, out_channels=3, num_blocks=[1, 1, 1], channels=32).to(device)
    criterion = HeritageRestorationLoss(ssim_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    start_epoch = 0
    best_score = -1.0
    patience_counter = 0
    history_rows = []
    
    # Resume checkpoint if requested
    if resume and os.path.exists(os.path.join(ckpt_dir, "latest.pth")):
        print(f"[RESUME] Loading latest checkpoint from {ckpt_dir}...")
        ckpt = load_checkpoint(model_name, device=device, load_latest=True)
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_score = ckpt.get("best_score", -1.0)
        
        hist_csv = os.path.join(ckpt_dir, "training_history.csv")
        if os.path.exists(hist_csv):
            history_rows = pd.read_csv(hist_csv).to_dict("records")
        print(f"[RESUME] Resuming from epoch {start_epoch+1}/{epochs} (Best Score: {best_score:.4f})")
    
    print(f"Starting {model_name} training for epochs {start_epoch+1} to {epochs} on {len(train_ds)} samples...")
    t_start = time.time()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            deg = batch["degraded"].to(device)
            clean = batch["clean"].to(device)
            
            optimizer.zero_grad()
            out = model(deg)
            loss = criterion(out, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss, psnr_list, ssim_list, lpips_list = 0.0, [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                deg = batch["degraded"].to(device)
                clean = batch["clean"].to(device)
                out = model(deg)
                val_loss += criterion(out, clean).item()
                
                out_c = torch.clamp(out, 0.0, 1.0)
                psnr_list.append(compute_psnr(out_c[0], clean[0]).item())
                
                out_np = (out_c[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                clean_np = (clean[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                ssim_list.append(compute_ssim(out_np, clean_np))
                if idx < 5:
                    lpips_list.append(compute_lpips(out_np, clean_np))
                    
        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))
        avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.33
        composite_score = compute_validation_composite_score(avg_psnr, avg_ssim, avg_lpips)
        cur_lr = optimizer.param_groups[0]['lr']
        
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
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | Composite: {composite_score:.4f}")
        
        config = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "channels": 32,
            "num_blocks": [1, 1, 1],
            "val_psnr": round(avg_psnr, 2),
            "val_ssim": round(avg_ssim, 4),
            "val_lpips": round(avg_lpips, 4),
            "composite_score": round(composite_score, 4)
        }
        
        # Check for improvement
        if composite_score > (best_score + 1e-4):
            best_score = composite_score
            patience_counter = 0
            save_checkpoint(model_name, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), epoch, best_score, config, is_best=True)
            print(f"  --> Saved new best checkpoint: Composite Score: {best_score:.4f} (PSNR: {avg_psnr:.2f} dB)")
        else:
            patience_counter += 1
            print(f"  --> No improvement for {patience_counter}/{patience} epochs.")
            
        save_checkpoint(model_name, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), epoch, composite_score, config, is_best=False)
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
    train_restormer(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                    resume=args.resume, force_retrain=args.force_retrain)
