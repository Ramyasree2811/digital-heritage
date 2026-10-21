import os
import cv2
import numpy as np
import torch

from models.super_resolution.model import SuperResolutionNet

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_super_resolution_model():
    """
    Loads the Super-Resolution model and its trained weights.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperResolutionNet()
    
    candidate_paths = [
        os.path.join(WORKSPACE_DIR, "outputs", "checkpoints", "superresolution", "best.pth"),
        os.path.join(WORKSPACE_DIR, "outputs", "super_resolution", "best_model.pth")
    ]
    loaded = False
    for weight_path in candidate_paths:
        if os.path.exists(weight_path):
            try:
                ckpt = torch.load(weight_path, map_location=device)
                state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
                model.load_state_dict(state)
                print(f"[SUPER-RES] Loaded Super-Resolution model weights from {weight_path}")
                loaded = True
                break
            except Exception as e:
                print(f"Warning: Could not load weights from {weight_path}: {e}")
    if not loaded:
        print("Notice: Super-Resolution checkpoint not found. Using initialized weights.")
        
    model.to(device)
    model.eval()
    return model, device

def upscale_image(image, scale_factor=2):
    """
    Upscales the restored image to a high resolution using the SR model.
    Falls back to high-quality bicubic interpolation and sharpening for crisp text.
    """
    h, w, c = image.shape
    
    # Get model and device
    model, device = get_super_resolution_model()
    
    # 1. Run inference
    try:
        # Preprocess input (normalize to [0, 1] and convert to tensor)
        img_arr = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            sr_tensor = model(img_tensor)
            
        sr_np = (sr_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        
        # Ensure correct target size (scale_factor * original size)
        target_h, target_w = h * scale_factor, w * scale_factor
        if sr_np.shape[:2] != (target_h, target_w):
            sr_np = cv2.resize(sr_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
        # Apply slight unsharp mask sharpening to enhance character edges
        blur = cv2.GaussianBlur(sr_np, (3, 3), 0)
        sr_np = cv2.addWeighted(sr_np, 1.3, blur, -0.3, 0)
        
        return sr_np
        
    except Exception as e:
        print(f"Warning: Super-Resolution model inference failed ({e}). Falling back to bicubic upscaling.")
        
        # High quality bicubic interpolation fallback
        target_h, target_w = h * scale_factor, w * scale_factor
        upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening to reduce bicubic blur
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
        upscaled = cv2.filter2D(upscaled, -1, kernel)
        
        return upscaled

if __name__ == "__main__":
    dummy = np.ones((64, 64, 3), dtype=np.uint8) * 200
    out = upscale_image(dummy)
    print("Super-Resolution upscaling test successful! Output shape:", out.shape)
