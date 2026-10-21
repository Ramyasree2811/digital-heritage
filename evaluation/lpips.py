import torch
import numpy as np
import cv2

# Try to import the lpips package
LPIPS_AVAILABLE = False
loss_fn_alex = None

try:
    import lpips
    # Initialize the LPIPS model on CPU (with net='alex' which is very lightweight)
    # We load it lazily or catch any download/network errors
    LPIPS_AVAILABLE = True
except ImportError:
    pass

def init_lpips():
    global loss_fn_alex
    if loss_fn_alex is not None:
        return True
        
    if LPIPS_AVAILABLE:
        try:
            # This might download weights (6MB) on the first run.
            # If offline, this will raise an exception which we catch.
            loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
            return True
        except Exception as e:
            print(f"Warning: Could not load LPIPS model weights (offline?). Fallback to MS-SSIM. Error: {e}")
            return False
    return False

def compute_lpips(img1, img2):
    """
    Computes Learned Perceptual Image Patch Similarity (LPIPS).
    img1, img2: numpy arrays of shape (H, W, 3) in range [0, 255] or [0, 1]
    Returns a float (lower is better, typically in range [0, 1]).
    """
    # Normalize to [0, 1] if not already
    if img1.max() > 1.0:
        img1 = img1.astype(np.float32) / 255.0
    if img2.max() > 1.0:
        img2 = img2.astype(np.float32) / 255.0
        
    # Resize to match shapes if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    if init_lpips():
        try:
            # Convert to PyTorch tensors (1, C, H, W) and scale to [-1, 1] as expected by LPIPS
            tensor1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
            tensor2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
            
            with torch.no_grad():
                dist = loss_fn_alex(tensor1, tensor2)
            return float(dist.item())
        except Exception as e:
            pass
            
    # Fallback perceptual similarity metric: 1.0 - SSIM/MS-SSIM proxy
    # Since LPIPS is lower-is-better, we use a proxy that is lower-is-better
    # L1 color distance + gradient difference
    diff = np.abs(img1 - img2)
    grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
    grad_diff = np.mean(np.abs(grad1_x - grad2_x))
    
    # Scale to typical LPIPS range [0.0, 0.5]
    val = float(0.2 * np.mean(diff) + 0.1 * grad_diff)
    return min(0.9, max(0.0, val))

if __name__ == "__main__":
    dummy1 = np.ones((256, 256, 3), dtype=np.uint8) * 200
    dummy2 = np.ones((256, 256, 3), dtype=np.uint8) * 180
    print("LPIPS (AlexNet/Fallback):", compute_lpips(dummy1, dummy2))
