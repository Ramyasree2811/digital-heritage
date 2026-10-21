import cv2
import numpy as np
import random

def draw_stain_mask(shape, center, radius, roughness=5):
    """
    Creates an irregular stain mask by drawing overlapping ellipses.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Base circle/ellipse
    cv2.ellipse(mask, center, (radius, int(radius * random.uniform(0.6, 1.0))), 
                random.randint(0, 360), 0, 360, 1.0, -1)
                
    # Add random bumps/sub-ellipses around the perimeter to make it irregular
    num_blobs = random.randint(3, 7)
    for _ in range(num_blobs):
        angle = random.uniform(0, 2 * np.pi)
        dist = radius * random.uniform(0.5, 0.9)
        blob_center = (
            int(center[0] + dist * np.cos(angle)),
            int(center[1] + dist * np.sin(angle))
        )
        blob_radius = int(radius * random.uniform(0.2, 0.5))
        cv2.ellipse(mask, blob_center, (blob_radius, int(blob_radius * random.uniform(0.5, 1.0))), 
                    random.randint(0, 360), 0, 360, 1.0, -1)
                    
    # Smooth the mask to create a soft, bled edge
    blur_kernel = int(radius * 0.4)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blur_kernel = max(3, blur_kernel)
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    return np.clip(mask, 0.0, 1.0)

def apply_stains(image, severity=0.5):
    """
    Applies water stains, mold, or ink spills to an RGB image.
    """
    if severity <= 0:
        return image
        
    stained_image = image.copy().astype(np.float32)
    h, w, c = image.shape
    
    num_stains = int(1 + severity * 4)  # 1 to 5 stains
    
    for _ in range(num_stains):
        # Choose stain type: water/humidity (brown/yellow), ink bleed (dark/blue), mold (grayish-green)
        stain_type = random.choice(["water", "water", "mold", "ink_bleed"])
        
        if stain_type == "water":
            # Brown/yellowish water stains
            color = np.array([random.randint(110, 160), random.randint(80, 120), random.randint(40, 70)], dtype=np.float32)
            alpha_multiplier = random.uniform(0.2, 0.4) * severity
        elif stain_type == "mold":
            # Greenish-gray mold/mildew
            color = np.array([random.randint(60, 90), random.randint(80, 110), random.randint(70, 95)], dtype=np.float32)
            alpha_multiplier = random.uniform(0.15, 0.3) * severity
        else:
            # Ink spill/bleed (dark blue/purple or black)
            color = random.choice([
                np.array([20, 20, 50], dtype=np.float32),  # Dark indigo
                np.array([10, 10, 15], dtype=np.float32)   # Carbon black
            ])
            alpha_multiplier = random.uniform(0.5, 0.8) * severity
            
        radius = random.randint(20, int(40 + severity * 80))
        center = (random.randint(radius, w - radius), random.randint(radius, h - radius))
        
        mask = draw_stain_mask(image.shape, center, radius)
        mask = np.expand_dims(mask, axis=2)  # (H, W, 1)
        
        # Linear blending: I_new = (1 - alpha*mask)*I_old + alpha*mask*color
        alpha = mask * alpha_multiplier
        stained_image = (1.0 - alpha) * stained_image + alpha * color
        
    return np.clip(stained_image, 0, 255).astype(np.uint8)
