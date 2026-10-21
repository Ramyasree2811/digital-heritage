import cv2
import numpy as np
import random

def add_wormholes(image, num_holes=3, max_radius=15, hole_color=(255, 255, 255)):
    """
    Simulates wormholes or paper decay by drawing solid circles with jagged edges.
    """
    h, w, c = image.shape
    decayed_image = image.copy()
    
    for _ in range(num_holes):
        center_x = random.randint(20, w - 20)
        center_y = random.randint(20, h - 20)
        radius = random.randint(4, max_radius)
        
        # Create an irregular circle mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Add tiny irregularities
        num_bumps = random.randint(2, 6)
        for _ in range(num_bumps):
            angle = random.uniform(0, 2 * np.pi)
            dist = radius * random.uniform(0.6, 1.0)
            bx = int(center_x + dist * np.cos(angle))
            by = int(center_y + dist * np.sin(angle))
            br = random.randint(2, int(radius * 0.6))
            cv2.circle(mask, (bx, by), br, 255, -1)
            
        decayed_image[mask == 255] = hole_color
        
    return decayed_image

def add_page_tears(image, corner_cut=True, edge_cut=True, background_color=(255, 255, 255)):
    """
    Simulates paper tear-offs on corners and edges.
    """
    h, w, c = image.shape
    teared_image = image.copy()
    
    # We will generate a mask of the teared regions
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Corner tear
    if corner_cut and random.random() < 0.7:
        # Choose which corner to tear (0: Top-Left, 1: Top-Right, 2: Bottom-Left, 3: Bottom-Right)
        corner = random.randint(0, 3)
        size_x = random.randint(15, w // 4)
        size_y = random.randint(15, h // 4)
        
        points = []
        if corner == 0:  # Top-Left
            points = [(0, size_y), (size_x, 0), (0, 0)]
        elif corner == 1:  # Top-Right
            points = [(w - 1 - size_x, 0), (w - 1, size_y), (w - 1, 0)]
        elif corner == 2:  # Bottom-Left
            points = [(0, h - 1 - size_y), (size_x, h - 1), (0, h - 1)]
        else:  # Bottom-Right
            points = [(w - 1 - size_x, h - 1), (w - 1, h - 1 - size_y), (w - 1, h - 1)]
            
        # Draw a jagged tear boundary
        # Instead of straight line between point 0 and point 1, make a jagged path
        p1, p2, p_corner = points
        jagged_path = [p1]
        steps = 10
        for i in range(1, steps):
            t = i / steps
            x_mid = int(p1[0] * (1 - t) + p2[0] * t + random.randint(-4, 4))
            y_mid = int(p1[1] * (1 - t) + p2[1] * t + random.randint(-4, 4))
            jagged_path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
        jagged_path.append(p2)
        jagged_path.append(p_corner)
        
        cv2.fillPoly(mask, [np.array(jagged_path, dtype=np.int32)], 255)
        
    # Edge tear (cutting out a chunk from one of the edges)
    if edge_cut and random.random() < 0.6:
        edge = random.randint(0, 3)  # 0: Top, 1: Bottom, 2: Left, 3: Right
        depth = random.randint(5, int(h * 0.08))
        width = random.randint(30, w // 3)
        
        if edge == 0:  # Top edge
            x_start = random.randint(10, w - 10 - width)
            points = [(x_start, 0), (x_start + width, 0)]
            # Draw jagged edge deep into the page
            path = [points[0]]
            steps = 8
            for i in range(1, steps):
                t = i / steps
                x_mid = int(points[0][0] * (1 - t) + points[1][0] * t)
                y_mid = int(depth * np.sin(t * np.pi) + random.randint(-3, 3))
                path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
            path.append(points[1])
            cv2.fillPoly(mask, [np.array(path, dtype=np.int32)], 255)
            
        elif edge == 1:  # Bottom edge
            x_start = random.randint(10, w - 10 - width)
            points = [(x_start, h - 1), (x_start + width, h - 1)]
            path = [points[0]]
            steps = 8
            for i in range(1, steps):
                t = i / steps
                x_mid = int(points[0][0] * (1 - t) + points[1][0] * t)
                y_mid = int((h - 1) - depth * np.sin(t * np.pi) + random.randint(-3, 3))
                path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
            path.append(points[1])
            cv2.fillPoly(mask, [np.array(path, dtype=np.int32)], 255)
            
        elif edge == 2:  # Left edge
            y_start = random.randint(10, h - 10 - width)
            points = [(0, y_start), (0, y_start + width)]
            path = [points[0]]
            steps = 8
            for i in range(1, steps):
                t = i / steps
                y_mid = int(points[0][1] * (1 - t) + points[1][1] * t)
                x_mid = int(depth * np.sin(t * np.pi) + random.randint(-3, 3))
                path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
            path.append(points[1])
            cv2.fillPoly(mask, [np.array(path, dtype=np.int32)], 255)
            
        else:  # Right edge
            y_start = random.randint(10, h - 10 - width)
            points = [(w - 1, y_start), (w - 1, y_start + width)]
            path = [points[0]]
            steps = 8
            for i in range(1, steps):
                t = i / steps
                y_mid = int(points[0][1] * (1 - t) + points[1][1] * t)
                x_mid = int((w - 1) - depth * np.sin(t * np.pi) + random.randint(-3, 3))
                path.append((np.clip(x_mid, 0, w - 1), np.clip(y_mid, 0, h - 1)))
            path.append(points[1])
            cv2.fillPoly(mask, [np.array(path, dtype=np.int32)], 255)
            
    teared_image[mask == 255] = background_color
    return teared_image

def apply_physical_defects(image, severity=0.5):
    """
    Applies page tears and wormholes to an RGB image.
    """
    if severity <= 0:
        return image
        
    defect_image = image.copy()
    
    # Paper tear-off background color: we can use a soft parchment white or solid white/black
    # White is standard for document scans
    bg_color = (255, 255, 255)
    
    # 1. Page Tears (edges/corners)
    if severity > 0.3:
        defect_image = add_page_tears(
            defect_image, 
            corner_cut=(severity > 0.5), 
            edge_cut=(severity > 0.3), 
            background_color=bg_color
        )
        
    # 2. Wormholes/Decay
    num_holes = int(severity * 4)  # 0 to 4 holes
    if num_holes > 0:
        max_radius = int(5 + severity * 15)  # Radius [5, 20]
        # Sometimes holes are filled with black (ink decay/burnt) or white (missing page scan background)
        hole_color = random.choice([(255, 255, 255), (0, 0, 0)])
        defect_image = add_wormholes(
            defect_image, 
            num_holes=num_holes, 
            max_radius=max_radius, 
            hole_color=hole_color
        )
        
    return defect_image
