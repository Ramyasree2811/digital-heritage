import cv2
import numpy as np
import random

def draw_single_fold(image, start_point, end_point, depth=1.5, thickness=1):
    """
    Draws a single 3D crease fold by pairing a dark shadow line and a light highlight line.
    """
    h, w, c = image.shape
    fold_img = image.copy().astype(np.float32)
    
    # Calculate perpendicular direction for the highlight offset
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return image
        
    nx = -dy / length
    ny = dx / length
    
    # Offset for shadow and highlight lines
    offset_shadow = 1.0
    offset_highlight = -1.0
    
    # Dark line (shadow)
    p_shadow_start = (int(round(start_point[0] + nx * offset_shadow)), int(round(start_point[1] + ny * offset_shadow)))
    p_shadow_end = (int(round(end_point[0] + nx * offset_shadow)), int(round(end_point[1] + ny * offset_shadow)))
    
    # Bright line (highlight)
    p_highlight_start = (int(round(start_point[0] + nx * offset_highlight)), int(round(start_point[1] + ny * offset_highlight)))
    p_highlight_end = (int(round(end_point[0] + nx * offset_highlight)), int(round(end_point[1] + ny * offset_highlight)))
    
    # Render shadow and highlight masks
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    highlight_mask = np.zeros((h, w), dtype=np.float32)
    
    cv2.line(shadow_mask, p_shadow_start, p_shadow_end, 1.0, thickness, cv2.LINE_AA)
    cv2.line(highlight_mask, p_highlight_start, p_highlight_end, 1.0, thickness, cv2.LINE_AA)
    
    # Blur masks to make the transitions smooth
    blur_kernel = 3
    shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_kernel, blur_kernel), 0)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (blur_kernel, blur_kernel), 0)
    
    shadow_mask = np.expand_dims(shadow_mask, axis=2)
    highlight_mask = np.expand_dims(highlight_mask, axis=2)
    
    # Apply shadow (darken) and highlight (brighten)
    # Shadow factor: darken by up to 20%
    shadow_factor = 0.20 * depth
    fold_img = fold_img * (1.0 - shadow_mask * shadow_factor)
    
    # Highlight factor: brighten by up to 25%
    highlight_factor = 0.25 * depth
    fold_img = fold_img + highlight_mask * highlight_factor * 255.0
    
    return np.clip(fold_img, 0, 255).astype(np.uint8)

def apply_folds(image, severity=0.5):
    """
    Applies multiple paper folds/creases to an RGB image.
    """
    if severity <= 0:
        return image
        
    h, w, c = image.shape
    folded_image = image.copy()
    
    # Number of folds
    num_folds = int(1 + severity * 4)  # 1 to 5 folds
    
    for _ in range(num_folds):
        # We can have full-page folds (crossing the page) or local wrinkles
        fold_type = random.choice(["full", "local"])
        
        if fold_type == "full":
            # Select start and end points on random boundaries
            side1 = random.randint(0, 3)
            side2 = (side1 + random.randint(1, 3)) % 4
            
            def get_point(side):
                if side == 0:  # Top
                    return (random.randint(0, w - 1), 0)
                elif side == 1:  # Bottom
                    return (random.randint(0, w - 1), h - 1)
                elif side == 2:  # Left
                    return (0, random.randint(0, h - 1))
                else:  # Right
                    return (w - 1, random.randint(0, h - 1))
                    
            start = get_point(side1)
            end = get_point(side2)
        else:
            # Local wrinkle (short fold inside the page)
            start = (random.randint(20, w - 20), random.randint(20, h - 20))
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(30, w // 3)
            end = (
                int(np.clip(start[0] + length * np.cos(angle), 0, w - 1)),
                int(np.clip(start[1] + length * np.sin(angle), 0, h - 1))
            )
            
        depth = random.uniform(0.6, 1.4) * severity
        thickness = random.choice([1, 2])
        
        folded_image = draw_single_fold(
            folded_image, 
            start, 
            end, 
            depth=depth, 
            thickness=thickness
        )
        
    return folded_image
