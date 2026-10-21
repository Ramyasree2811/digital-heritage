import cv2
import numpy as np
import random

def add_gaussian_noise(image, mean=0, sigma=20):
    image = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, amount=0.015, salt_vs_pepper=0.5):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    num_noise = int(amount * h * w)
    
    # Salt (white pixels)
    num_salt = int(num_noise * salt_vs_pepper)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in [h, w]]
    noisy_image[coords_salt[0], coords_salt[1], :] = 255
    
    # Pepper (black pixels)
    num_pepper = int(num_noise * (1.0 - salt_vs_pepper))
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in [h, w]]
    noisy_image[coords_pepper[0], coords_pepper[1], :] = 0
    
    return noisy_image

def apply_blur(image, blur_type="gaussian", size=5):
    if size % 2 == 0:
        size += 1
    size = max(3, size)
    
    if blur_type == "gaussian":
        return cv2.GaussianBlur(image, (size, size), 0)
    elif blur_type == "motion":
        kernel = np.zeros((size, size))
        angle = random.randint(0, 360)
        rad = np.deg2rad(angle)
        dx, dy = np.cos(rad), np.sin(rad)
        center = size // 2
        for i in range(size):
            offset = i - center
            x = int(round(center + offset * dx))
            y = int(round(center + offset * dy))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel)
    return image

def draw_scratches(image, mask, num_scratches=5, max_length=100, max_thickness=2):
    h, w, c = image.shape
    degraded = image.copy()
    
    for _ in range(num_scratches):
        x, y = random.randint(10, w - 10), random.randint(10, h - 10)
        length = random.randint(20, max_length)
        thickness = random.randint(1, max_thickness)
        
        # Heritage scratch colors (white for scratches, or black/brown)
        color = random.choice([(235, 235, 235), (20, 20, 20), (60, 45, 30)])
        
        points = [(x, y)]
        angle = random.uniform(0, 2 * np.pi)
        
        for _ in range(length // 4):
            angle += random.uniform(-0.3, 0.3)
            dx = int(round(np.cos(angle) * 4))
            dy = int(round(np.sin(angle) * 4))
            x = np.clip(x + dx, 0, w - 1)
            y = np.clip(y + dy, 0, h - 1)
            if x == points[-1][0] and y == points[-1][1]:
                break
            points.append((x, y))
            
        for i in range(len(points) - 1):
            cv2.line(degraded, points[i], points[i+1], color, thickness, cv2.LINE_AA)
            cv2.line(mask, points[i], points[i+1], 255, thickness, cv2.LINE_AA)
            
    return degraded

def draw_stains(image, mask, num_stains=3, max_radius=40):
    h, w, c = image.shape
    degraded = image.copy().astype(np.float32)
    
    for _ in range(num_stains):
        center = (random.randint(20, w - 20), random.randint(20, h - 20))
        radius = random.randint(15, max_radius)
        
        # Stain shape
        stain_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(stain_mask, center, radius, 1.0, -1)
        
        # Add random irregularities
        num_bumps = random.randint(3, 6)
        for _ in range(num_bumps):
            angle = random.uniform(0, 2 * np.pi)
            dist = radius * random.uniform(0.6, 0.9)
            bx = int(center[0] + dist * np.cos(angle))
            by = int(center[1] + dist * np.sin(angle))
            br = random.randint(5, int(radius * 0.5))
            cv2.circle(stain_mask, (bx, by), br, 1.0, -1)
            
        # Soften stain edges
        blur_kernel = int(radius * 0.5) | 1  # make odd
        stain_mask = cv2.GaussianBlur(stain_mask, (blur_kernel, blur_kernel), 0)
        stain_mask = np.clip(stain_mask, 0.0, 1.0)
        
        # Water/grease brown color
        color = np.array([random.randint(100, 150), random.randint(70, 110), random.randint(40, 70)], dtype=np.float32)
        alpha = np.expand_dims(stain_mask, axis=2) * random.uniform(0.2, 0.5)
        
        degraded = (1.0 - alpha) * degraded + alpha * color
        
        # Update damage mask where stain is visible
        mask[stain_mask > 0.1] = 255
        
    return np.clip(degraded, 0, 255).astype(np.uint8)

def apply_general_degradation(image, severity=0.5):
    """
    Applies noise, blur, scratches, and stains to an RGB image.
    Returns: (degraded_image, damage_mask)
    """
    h, w, c = image.shape
    degraded = image.copy()
    damage_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. Stains
    if random.random() < 0.6:
        num_stains = int(1 + severity * 4)
        max_r = int(20 + severity * 40)
        degraded = draw_stains(degraded, damage_mask, num_stains=num_stains, max_radius=max_r)
        
    # 2. Scratches
    if random.random() < 0.7:
        num_sc = int(1 + severity * 8)
        max_l = int(40 + severity * 120)
        degraded = draw_scratches(degraded, damage_mask, num_scratches=num_sc, max_length=max_l)
        
    # 3. Blur (Global defect, doesn't add to localized damage mask)
    if random.random() < 0.5:
        size = int(3 + severity * 8)
        b_type = random.choice(["gaussian", "motion"])
        degraded = apply_blur(degraded, blur_type=b_type, size=size)
        
    # 4. Noise (Global defect, doesn't add to localized damage mask)
    if random.random() < 0.5:
        n_type = random.choice(["gaussian", "salt_and_pepper"])
        if n_type == "gaussian":
            sigma = int(5 + severity * 30)
            degraded = add_gaussian_noise(degraded, sigma=sigma)
        else:
            amount = 0.005 + severity * 0.03
            degraded = add_salt_and_pepper_noise(degraded, amount=amount)
            # Add salt & pepper dots to damage mask
            # Detect black/white changes
            diff = cv2.absdiff(degraded, image)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            damage_mask[diff_gray > 200] = 255
            
    return degraded, damage_mask

if __name__ == "__main__":
    dummy = np.ones((256, 256, 3), dtype=np.uint8) * 200
    deg, mask = apply_general_degradation(dummy, severity=0.6)
    print("General degradation test successful!")
    print("Degraded shape:", deg.shape, "Mask shape:", mask.shape, "Mask white pixels:", np.sum(mask == 255))
