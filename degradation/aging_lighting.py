import cv2
import numpy as np
import random

def apply_fading_and_yellowing(image, severity=0.5):
    """
    Simulates paper aging (yellowing/sepia shift) and ink fading (contrast reduction).
    """
    h, w, c = image.shape
    aged_img = image.astype(np.float32)
    
    # 1. Color shift to yellow-brown (Sepia/aging effect)
    # R gets boosted, G gets slightly boosted, B gets reduced
    # For a realistic effect, shift B downwards and R upwards
    shift_b = 30 * severity
    shift_g = 5 * severity
    shift_r = 15 * severity
    
    aged_img[:, :, 0] = np.clip(aged_img[:, :, 0] + shift_r, 0, 255)  # R channel
    aged_img[:, :, 1] = np.clip(aged_img[:, :, 1] + shift_g, 0, 255)  # G channel
    aged_img[:, :, 2] = np.clip(aged_img[:, :, 2] - shift_b, 0, 255)  # B channel
    
    # 2. Contrast fading (reducing difference between text and paper)
    # Pull darks up, pull brights down
    # x_faded = x * (1 - alpha) + alpha * neutral_grey
    # Or shift blacks (darks) upwards and whites (brights) downwards
    fade_factor = 0.1 + 0.3 * severity  # range [0.1, 0.4]
    
    # Simple linear contrast compression
    mean_val = 180.0  # historical documents typically have bright paper background
    aged_img = aged_img * (1.0 - fade_factor) + mean_val * fade_factor
    
    return np.clip(aged_img, 0, 255).astype(np.uint8)

def apply_uneven_lighting(image, severity=0.5):
    """
    Simulates shadows, lighting gradients, and flash hotspots.
    """
    h, w, c = image.shape
    lit_img = image.astype(np.float32)
    
    # Choose between a linear shadow gradient (edge shadow) or a radial shadow (corner vignetting)
    light_type = random.choice(["linear", "radial"])
    
    light_map = np.ones((h, w), dtype=np.float32)
    
    if light_type == "linear":
        # Linear gradient at a random angle
        angle = random.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Min light value represents how dark the shadow is (lower = darker)
        min_light = 0.9 - 0.45 * severity  # range [0.45, 0.90]
        
        for y in range(h):
            for x in range(w):
                # Project point onto direction vector and normalize
                projection = (x * dx + y * dy)
                light_map[y, x] = projection
                
        # Normalize light map to range [min_light, 1.0]
        l_min, l_max = light_map.min(), light_map.max()
        if l_max != l_min:
            light_map = min_light + (1.0 - min_light) * (light_map - l_min) / (l_max - l_min)
        else:
            light_map = np.ones((h, w), dtype=np.float32)
            
    else:
        # Radial gradient (vignette / center spotlight)
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)
        max_dist = np.sqrt(max(center_x, w - center_x)**2 + max(center_y, h - center_y)**2)
        
        min_light = 0.85 - 0.4 * severity  # range [0.45, 0.85]
        
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                # Light decreases quadratically/linearly away from center
                t = min(1.0, dist / max_dist)
                light_map[y, x] = 1.0 - (1.0 - min_light) * t
                
    # Apply lighting map to image
    light_map = np.expand_dims(light_map, axis=2)
    lit_img = lit_img * light_map
    
    return np.clip(lit_img, 0, 255).astype(np.uint8)

def apply_aging_lighting(image, severity=0.5):
    """
    Applies paper yellowing, fading, and lighting gradients to an RGB image.
    """
    if severity <= 0:
        return image
        
    aged_image = image.copy()
    
    # Apply fading and aging
    if random.random() < 0.8:
        aged_image = apply_fading_and_yellowing(aged_image, severity=severity)
        
    # Apply uneven lighting shadows
    if random.random() < 0.8:
        aged_image = apply_uneven_lighting(aged_image, severity=severity)
        
    return aged_image
