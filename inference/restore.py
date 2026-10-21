import os
import sys

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

import json
import cv2
import numpy as np
import torch

from models.unet.model import UNet
from models.dncnn.model import DnCNN
from models.pix2pix.model import Pix2PixGenerator
from models.restormer.model import Restormer
from models.swinir.model import SwinIR

OUTPUTS_DIR = os.path.join(WORKSPACE_DIR, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(OUTPUTS_DIR, "best_model")

def get_restoration_model(model_name="best"):
    """
    Instantiates the selected model and loads its trained checkpoint.
    If model_name is "best" or "auto", loads from outputs/best_model/model.pth.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean_name = model_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    
    # 1. Handle "best" or "auto" model
    if clean_name in ["best", "auto", "bestmodel"]:
        best_cfg_path = os.path.join(BEST_MODEL_DIR, "model_config.json")
        best_weight_path = os.path.join(BEST_MODEL_DIR, "model.pth")
        
        target_name = "UNet"
        if os.path.exists(best_cfg_path):
            try:
                with open(best_cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    target_name = cfg.get("model_name", "UNet")
            except Exception:
                pass
                
        model, _ = get_restoration_model(target_name)
        if os.path.exists(best_weight_path):
            try:
                ckpt = torch.load(best_weight_path, map_location=device)
                state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
                model.load_state_dict(state)
                print(f"[RESTORE] Loaded deployed Best Model ({target_name}) from {best_weight_path}")
                return model, device
            except Exception as e:
                print(f"Warning: Could not load best_model weights ({e}).")
        return model, device

    # 2. Instantiate specific architecture
    if clean_name == "unet":
        model = UNet()
        candidate_paths = [
            os.path.join(CHECKPOINTS_DIR, "unet", "best.pth"),
            os.path.join(WORKSPACE_DIR, "outputs", "unet", "best_model.pth")
        ]
    elif clean_name == "dncnn":
        model = DnCNN(in_channels=3, out_channels=3, num_layers=8, num_features=32)
        candidate_paths = [
            os.path.join(CHECKPOINTS_DIR, "dncnn", "best.pth"),
            os.path.join(WORKSPACE_DIR, "outputs", "dncnn", "best_model.pth")
        ]
    elif clean_name in ["pix2pix", "pix2pixgan"]:
        model = Pix2PixGenerator()
        candidate_paths = [
            os.path.join(CHECKPOINTS_DIR, "pix2pix", "best_generator.pth"),
            os.path.join(WORKSPACE_DIR, "outputs", "pix2pix", "best_model.pth")
        ]
    elif clean_name == "restormer":
        model = Restormer(in_channels=3, out_channels=3, num_blocks=[1, 1, 1], channels=32)
        candidate_paths = [
            os.path.join(CHECKPOINTS_DIR, "restormer", "best.pth"),
            os.path.join(WORKSPACE_DIR, "outputs", "restormer", "best_model.pth")
        ]
    elif clean_name == "swinir":
        model = SwinIR(in_channels=3, out_channels=3, img_size=256, channels=32, num_heads=4)
        candidate_paths = [
            os.path.join(CHECKPOINTS_DIR, "swinir", "best.pth"),
            os.path.join(WORKSPACE_DIR, "outputs", "swinir", "best_model.pth")
        ]
    else:
        # Fallback to UNet
        model = UNet()
        candidate_paths = [os.path.join(CHECKPOINTS_DIR, "unet", "best.pth")]

    loaded = False
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location=device)
                state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
                model.load_state_dict(state)
                print(f"[RESTORE] Loaded {model_name} weights from {p}")
                loaded = True
                break
            except Exception as e:
                print(f"Warning: Could not load weights from {p}: {e}")
                
    if not loaded:
        print(f"Notice: Running {model_name} with default initialization (no trained checkpoint found).")

    model.to(device)
    model.eval()
    return model, device

def apply_post_processing(original_img, restored_img, is_document=True, damage_mask=None):
    """
    Applies high-fidelity heritage image enhancement:
    - Never erases text strokes
    - Normalizes background illumination, removing aging stains, foxing, and shadows
    - Generates pristine, high-contrast, clean-background restoration matching archival standards
    """
    if restored_img.max() <= 1.0:
        restored_img = (restored_img * 255.0).astype(np.uint8)
    if original_img.max() <= 1.0:
        original_img = (original_img * 255.0).astype(np.uint8)
        
    h, w = original_img.shape[:2]
    if restored_img.shape[:2] != (h, w):
        restored_img = cv2.resize(restored_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    cleaned = restored_img.copy()

    if is_document:
        # Convert to LAB space to independently process lightness and authentic chroma
        lab = cv2.cvtColor(cleaned, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        
        # 1. Macro-illumination leveling (large kernel so text strokes are never altered)
        k_size = max(35, (min(h, w) // 4) | 1)
        bg = cv2.morphologyEx(l_chan, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)))
        bg = cv2.medianBlur(bg, max(21, (k_size // 2) * 2 + 1))
        norm_l = np.clip((l_chan.astype(np.float32) / (bg.astype(np.float32) + 1e-4)) * 236.0, 0, 255).astype(np.uint8)
        
        # 2. Bilateral edge-preserving filter: eliminates camera noise/speckles while preserving crisp character stroke boundaries
        denoised_l = cv2.bilateralFilter(norm_l, d=5, sigmaColor=35, sigmaSpace=35)
        
        # 3. Selective continuous ink stroke deepening (NO binary speckles or adaptive thresholding)
        ink_mask = np.clip((215.0 - denoised_l.astype(np.float32)) / 75.0, 0.0, 1.0)
        enhanced_l = denoised_l.astype(np.float32) - (ink_mask * 28.0)
        enhanced_l = np.clip(enhanced_l, 0, 255).astype(np.uint8)
        
        # 4. Gentle unsharp masking for stroke crispness
        blurred = cv2.GaussianBlur(enhanced_l, (0, 0), 1.0)
        sharp_l = cv2.addWeighted(enhanced_l, 1.25, blurred, -0.25, 0)
        sharp_l = np.clip(sharp_l, 0, 255).astype(np.uint8)
        
        # 5. Chroma smoothing to remove color grain while retaining authentic aged parchment/palm-leaf hue
        denoised_a = cv2.bilateralFilter(a_chan, d=5, sigmaColor=20, sigmaSpace=20)
        denoised_b = cv2.bilateralFilter(b_chan, d=5, sigmaColor=20, sigmaSpace=20)
        
        cleaned = cv2.cvtColor(cv2.merge((sharp_l, denoised_a, denoised_b)), cv2.COLOR_LAB2RGB)

    else:
        # Monuments & Sculptures enhancement
        lab = cv2.cvtColor(cleaned, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_sharp = clahe.apply(l)
        cleaned = cv2.cvtColor(cv2.merge((l_sharp, a, b)), cv2.COLOR_LAB2RGB)

        # Structural high-frequency relief fusion
        blurred_orig = cv2.GaussianBlur(original_img, (3, 3), 0.8)
        high_freq = original_img.astype(np.float32) - blurred_orig.astype(np.float32)
        cleaned = np.clip(cleaned.astype(np.float32) + 0.30 * high_freq, 0, 255).astype(np.uint8)

    return cleaned

def restore_patch(model, patch_np, device):
    """Runs model on a single 256x256 RGB patch."""
    patch_arr = patch_np.astype(np.float32) / 255.0
    patch_t = torch.from_numpy(patch_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out_t = model(patch_t)
    out_np = (torch.clamp(out_t[0], 0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return out_np

def restore_heritage_image(image, model_name="best", damage_mask=None, is_document=True):
    """
    Full restoration engine with native resolution support and edge-guided blending.
    image: numpy array (H, W, 3) RGB
    """
    h, w, c = image.shape
    model, device = get_restoration_model(model_name)
    
    # Scale down extreme megapixel photos (e.g. 4K camera uploads) to max 1200px for speed
    working_image = image.copy()
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        working_image = cv2.resize(working_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h_work, w_work = working_image.shape[:2]
    else:
        h_work, w_work = h, w

    # For DnCNN: Fully convolutional architecture, can process directly at native resolution!
    is_dncnn = isinstance(model, DnCNN) or ("dncnn" in model_name.lower())
    if is_dncnn:
        pad_h = (16 - (h_work % 16)) % 16
        pad_w = (16 - (w_work % 16)) % 16
        padded = cv2.copyMakeBorder(working_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        t = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out_t = model(t)
        out_np = (torch.clamp(out_t[0], 0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        restored_raw = out_np[:h_work, :w_work]
        
        # Scale back to original resolution if downscaled
        if (h_work, w_work) != (h, w):
            restored_raw = cv2.resize(restored_raw, (w, h), interpolation=cv2.INTER_CUBIC)
            
        return apply_post_processing(image, restored_raw, is_document=is_document, damage_mask=damage_mask)

    # For other models (e.g. SwinIR, Restormer, UNet):
    # Use tiled sliding-window inference with Hann window blending at native resolution (no destructive 256 downscaling)
    patch_size = 256
    stride = 192  # 64px overlap
    
    # If working image is small enough, pad to 256
    if h_work <= patch_size and w_work <= patch_size:
        pad_h = patch_size - h_work
        pad_w = patch_size - w_work
        padded = cv2.copyMakeBorder(working_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        restored_patch = restore_patch(model, padded, device)
        restored_raw = restored_patch[:h_work, :w_work]
        if (h_work, w_work) != (h, w):
            restored_raw = cv2.resize(restored_raw, (w, h), interpolation=cv2.INTER_CUBIC)
        return apply_post_processing(image, restored_raw, is_document=is_document, damage_mask=damage_mask)

    # Tiled inference for larger images
    y_win = np.hanning(patch_size)
    x_win = np.hanning(patch_size)
    blend_weight = np.outer(y_win, x_win)
    blend_weight = np.expand_dims(blend_weight, axis=-1)

    canvas = np.zeros((h_work, w_work, 3), dtype=np.float32)
    weight_canvas = np.zeros((h_work, w_work, 1), dtype=np.float32) + 1e-6

    y_steps = list(range(0, h_work - patch_size, stride))
    if not y_steps or y_steps[-1] != h_work - patch_size:
        y_steps.append(max(0, h_work - patch_size))
        
    x_steps = list(range(0, w_work - patch_size, stride))
    if not x_steps or x_steps[-1] != w_work - patch_size:
        x_steps.append(max(0, w_work - patch_size))

    for y in y_steps:
        for x in x_steps:
            patch = working_image[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            restored_patch = restore_patch(model, patch, device)
            canvas[y:y+patch_size, x:x+patch_size] += restored_patch.astype(np.float32) * blend_weight
            weight_canvas[y:y+patch_size, x:x+patch_size] += blend_weight

    blended = np.clip(canvas / weight_canvas, 0, 255).astype(np.uint8)
    if (h_work, w_work) != (h, w):
        blended = cv2.resize(blended, (w, h), interpolation=cv2.INTER_CUBIC)
        
    return apply_post_processing(image, blended, is_document=is_document, damage_mask=damage_mask)

if __name__ == "__main__":
    dummy = np.ones((512, 512, 3), dtype=np.uint8) * 180
    res = restore_heritage_image(dummy, model_name="best")
    print("Restore engine test complete! Output shape:", res.shape)
