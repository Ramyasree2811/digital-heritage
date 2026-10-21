import cv2
import numpy as np
import random

def draw_cracks(image, mask, num_cracks=3, max_length=150, thickness=2):
    """
    Draws dark, thin, sharp jagged lines simulating splits/cracks in a palm leaf.
    """
    h, w, c = image.shape
    degraded = image.copy()
    
    for _ in range(num_cracks):
        # Starts near the edge or inside
        x = random.choice([random.randint(0, w // 10), random.randint(9 * w // 10, w - 1), random.randint(20, w - 20)])
        y = random.choice([random.randint(0, h // 10), random.randint(9 * h // 10, h - 1), random.randint(20, h - 20)])
        
        length = random.randint(30, max_length)
        # Crack color: dark brown/black
        color = (15, 12, 10)
        
        points = [(x, y)]
        # Favour horizontal or vertical depending on initialization
        angle = random.choice([0, np.pi]) if random.random() < 0.7 else random.choice([np.pi/2, 3*np.pi/2])
        
        for _ in range(length // 5):
            angle += random.uniform(-0.15, 0.15)
            dx = int(round(np.cos(angle) * 5))
            dy = int(round(np.sin(angle) * 5))
            x = np.clip(x + dx, 0, w - 1)
            y = np.clip(y + dy, 0, h - 1)
            
            if x == points[-1][0] and y == points[-1][1]:
                break
            points.append((x, y))
            
        for i in range(len(points) - 1):
            cv2.line(degraded, points[i], points[i+1], color, thickness, cv2.LINE_AA)
            cv2.line(mask, points[i], points[i+1], 255, thickness, cv2.LINE_AA)
            
    return degraded

def draw_wormholes(image, mask, num_holes=3, max_radius=12, hole_color=(255, 255, 255)):
    """
    Simulates wormholes or physical leaf decay by drawing solid blobs.
    """
    h, w, c = image.shape
    degraded = image.copy()
    
    for _ in range(num_holes):
        center = (random.randint(30, w - 30), random.randint(30, h - 30))
        radius = random.randint(3, max_radius)
        
        # Create irregular blob mask
        blob_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(blob_mask, center, radius, 255, -1)
        
        num_bumps = random.randint(2, 5)
        for _ in range(num_bumps):
            angle = random.uniform(0, 2 * np.pi)
            dist = radius * random.uniform(0.5, 0.9)
            bx = int(center[0] + dist * np.cos(angle))
            by = int(center[1] + dist * np.sin(angle))
            br = random.randint(1, max(1, int(radius * 0.5)))
            cv2.circle(blob_mask, (bx, by), br, 255, -1)
            
        degraded[blob_mask == 255] = hole_color
        mask[blob_mask == 255] = 255
        
    return degraded

def apply_broken_edges(image, mask, depth=20, bg_color=(255, 255, 255)):
    """
    Cuts off jagged chunks from palm-leaf boundaries.
    """
    h, w, c = image.shape
    degraded = image.copy()
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 0: Top edge, 1: Bottom edge, 2: Left edge, 3: Right edge
    edges_to_tear = random.sample([0, 1, 2, 3], random.randint(1, 2))
    
    for edge in edges_to_tear:
        if edge == 0:  # Top
            x_start = random.randint(0, w // 2)
            width = random.randint(40, w - x_start)
            points = [(x_start, 0), (x_start + width, 0)]
            path = [points[0]]
            steps = 10
            for i in range(1, steps):
                t = i / steps
                x_mid = int(points[0][0] * (1-t) + points[1][0] * t)
                y_mid = int(depth * np.sin(t * np.pi) + random.randint(-2, 2))
                path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
            path.append(points[1])
            cv2.fillPoly(edge_mask, [np.array(path, dtype=np.int32)], 255)
            
        elif edge == 1:  # Bottom
            x_start = random.randint(0, w // 2)
            width = random.randint(40, w - x_start)
            points = [(x_start, h - 1), (x_start + width, h - 1)]
            path = [points[0]]
            steps = 10
            for i in range(1, steps):
                t = i / steps
                x_mid = int(points[0][0] * (1-t) + points[1][0] * t)
                y_mid = int((h - 1) - depth * np.sin(t * np.pi) + random.randint(-2, 2))
                path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
            path.append(points[1])
            cv2.fillPoly(edge_mask, [np.array(path, dtype=np.int32)], 255)
            
        elif edge == 2:  # Left
            y_start = random.randint(0, h // 2)
            width = random.randint(40, h - y_start)
            points = [(0, y_start), (0, y_start + width)]
            path = [points[0]]
            steps = 10
            for i in range(1, steps):
                t = i / steps
                y_mid = int(points[0][1] * (1-t) + points[1][1] * t)
                x_mid = int(depth * np.sin(t * np.pi) + random.randint(-2, 2))
                path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
            path.append(points[1])
            cv2.fillPoly(edge_mask, [np.array(path, dtype=np.int32)], 255)
            
        else:  # Right
            y_start = random.randint(0, h // 2)
            width = random.randint(40, h - y_start)
            points = [(w - 1, y_start), (w - 1, y_start + width)]
            path = [points[0]]
            steps = 10
            for i in range(1, steps):
                t = i / steps
                y_mid = int(points[0][1] * (1-t) + points[1][1] * t)
                x_mid = int((w - 1) - depth * np.sin(t * np.pi) + random.randint(-2, 2))
                path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
            path.append(points[1])
            cv2.fillPoly(edge_mask, [np.array(path, dtype=np.int32)], 255)
            
    degraded[edge_mask == 255] = bg_color
    mask[edge_mask == 255] = 255
    return degraded

def apply_character_fading(image, mask, severity=0.5):
    """
    Fades dark text characters in parts of the image to simulate worn-out ink.
    """
    h, w, c = image.shape
    faded = image.astype(np.float32)
    
    # Generate spatial fading regions (circles of fading)
    fade_mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(random.randint(1, 3)):
        center = (random.randint(20, w - 20), random.randint(20, h - 20))
        radius = random.randint(40, 100)
        cv2.circle(fade_mask, center, radius, 1.0, -1)
        
    fade_mask = cv2.GaussianBlur(fade_mask, (45, 45), 0)
    fade_mask = np.expand_dims(fade_mask * severity * 0.7, axis=2)  # Max 70% fade
    
    # Background color: warm brownish/yellowish leaf background
    leaf_bg = np.array([210, 180, 130], dtype=np.float32)  # warm palm-leaf color
    
    # We only want to fade pixels that are darker (the text)
    # Convert image to grayscale to find dark text strokes
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    text_mask = np.expand_dims((gray < 110).astype(np.float32), axis=2)
    
    # Combine: where text is present and fading is active, blend text closer to background
    # Faded_val = (1 - fade)*original + fade*bg
    faded_blend = text_mask * fade_mask
    faded = (1.0 - faded_blend) * faded + faded_blend * leaf_bg
    
    # Add faded areas to the damage mask
    # Find where text changed significantly
    diff = cv2.absdiff(faded.astype(np.uint8), image)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    mask[diff_gray > 25] = 255
    
    return np.clip(faded, 0, 255).astype(np.uint8)

def apply_uneven_lighting(image, severity=0.5):
    """
    Applies radial vignetting/lighting drop to simulate camera/environment shadow.
    """
    h, w, c = image.shape
    lit = image.astype(np.float32)
    
    center_x = random.randint(w // 3, 2 * w // 3)
    center_y = random.randint(h // 3, 2 * h // 3)
    max_dist = np.sqrt(max(center_x, w - center_x)**2 + max(center_y, h - center_y)**2)
    
    min_light = 0.85 - 0.45 * severity  # [0.4, 0.85]
    
    # Optimized grid calculation
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    light_map = 1.0 - (1.0 - min_light) * (dist / max_dist)
    light_map = np.expand_dims(np.clip(light_map, min_light, 1.0), axis=2)
    
    lit = lit * light_map
    return np.clip(lit, 0, 255).astype(np.uint8)

def apply_palm_leaf_degradation(image, severity=0.5):
    """
    Applies specialized palm-leaf degradation: cracks, holes, edge tear-offs,
    mold stains, character fading, and uneven lighting.
    Returns: (degraded_image, damage_mask)
    """
    h, w, c = image.shape
    degraded = image.copy()
    damage_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. Edge Tears / Broken borders
    if random.random() < 0.6:
        depth = int(5 + severity * 20)
        # Background shelf color of digital archives is usually black or white
        bg_c = random.choice([(255, 255, 255), (10, 10, 10)])
        degraded = apply_broken_edges(degraded, damage_mask, depth=depth, bg_color=bg_c)
        
    # 2. Wormholes / Leaf decay holes
    if random.random() < 0.7:
        num_holes = int(1 + severity * 5)
        max_r = int(4 + severity * 12)
        # Holes can show black background underneath the leaf or white scan page
        h_color = random.choice([(255, 255, 255), (15, 15, 15)])
        
        # We need to define drawing wormholes here to avoid references to undefined variables
        # (The inline definition above had a typo "decayed" instead of "degraded", let's fix it by defining it correctly)
        for _ in range(num_holes):
            center = (random.randint(30, w - 30), random.randint(30, h - 30))
            radius = random.randint(3, max_r)
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(blob_mask, center, radius, 255, -1)
            num_bumps = random.randint(2, 5)
            for _ in range(num_bumps):
                angle = random.uniform(0, 2 * np.pi)
                dist = radius * random.uniform(0.5, 0.9)
                bx = int(center[0] + dist * np.cos(angle))
                by = int(center[1] + dist * np.sin(angle))
                br = random.randint(1, max(1, int(radius * 0.5)))
                cv2.circle(blob_mask, (bx, by), br, 255, -1)
            degraded[blob_mask == 255] = h_color
            damage_mask[blob_mask == 255] = 255
            
    # 3. Cracks / Splits
    if random.random() < 0.6:
        num_cracks = int(1 + severity * 4)
        max_l = int(50 + severity * 150)
        thickness = 1 if severity < 0.5 else 2
        degraded = draw_cracks(degraded, damage_mask, num_cracks=num_cracks, max_length=max_l, thickness=thickness)
        
    # 4. Character Fading / Ink wearing
    if random.random() < 0.8:
        degraded = apply_character_fading(degraded, damage_mask, severity=severity)
        
    # 5. Mildew / Dark mold stains
    if random.random() < 0.5:
        # Small, dark mold spots
        num_stains = int(2 + severity * 6)
        for _ in range(num_stains):
            center = (random.randint(15, w - 15), random.randint(15, h - 15))
            r = random.randint(3, 15)
            stain_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(stain_mask, center, r, 1.0, -1)
            stain_mask = cv2.GaussianBlur(stain_mask, (9, 9), 0)
            stain_mask = np.expand_dims(np.clip(stain_mask, 0.0, 1.0), axis=2)
            
            # Dark greyish mold color
            mold_c = np.array([random.randint(40, 70), random.randint(45, 75), random.randint(40, 70)], dtype=np.float32)
            alpha = stain_mask * random.uniform(0.3, 0.6) * severity
            degraded = (1.0 - alpha) * degraded.astype(np.float32) + alpha * mold_c
            degraded = np.clip(degraded, 0, 255).astype(np.uint8)
            
            # Update mask where mold is concentrated
            damage_mask[stain_mask.squeeze(-1) > 0.2] = 255
            
    # 6. Uneven lighting
    if random.random() < 0.7:
        degraded = apply_uneven_lighting(degraded, severity=severity)
        
    return degraded, damage_mask

if __name__ == "__main__":
    dummy = np.ones((256, 256, 3), dtype=np.uint8) * 200
    cv2.putText(dummy, "Devanagari Test", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    deg, mask = apply_palm_leaf_degradation(dummy, severity=0.6)
    print("Palm leaf degradation test successful! Mask pixels:", np.sum(mask == 255))
