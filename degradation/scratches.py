import cv2
import numpy as np
import random

def draw_random_walk_scratch(image, start_point=None, length=50, thickness=1, color=None):
    """
    Draws a single scratch using a random walk with momentum to simulate real surface scratches.
    """
    h, w, c = image.shape
    scratch_img = image.copy()
    
    if start_point is None:
        start_point = (random.randint(10, w - 10), random.randint(10, h - 10))
        
    if color is None:
        # Heritage scratches can be dark (ink scratches) or light (surface paper scrapes)
        color = random.choice([
            (200, 200, 200),  # Light grey
            (240, 240, 240),  # Off-white
            (50, 40, 30),     # Dark brown
            (10, 10, 10)      # Black/dark ink
        ])
        
    x, y = start_point
    angle = random.uniform(0, 2 * np.pi)
    momentum = 0.8
    
    points = [(x, y)]
    for _ in range(length):
        # Update angle with some random noise but keep momentum
        angle += random.uniform(-0.2, 0.2)
        dx = int(round(np.cos(angle) * 3))
        dy = int(round(np.sin(angle) * 3))
        
        x = np.clip(x + dx, 0, w - 1)
        y = np.clip(y + dy, 0, h - 1)
        
        # Avoid looping at boundaries
        if x == points[-1][0] and y == points[-1][1]:
            break
            
        points.append((x, y))
        
    # Draw scratch lines
    for i in range(len(points) - 1):
        cv2.line(scratch_img, points[i], points[i+1], color, thickness, cv2.LINE_AA)
        
    return scratch_img

def apply_scratches(image, severity=0.5):
    """
    Applies multiple random walk scratches to an RGB image.
    """
    if severity <= 0:
        return image
        
    num_scratches = int(1 + severity * 12)  # 1 to 13 scratches
    scratched_image = image.copy()
    
    for _ in range(num_scratches):
        length = random.randint(30, int(100 + severity * 150))
        thickness = random.choice([1, 1, 2]) if severity < 0.7 else random.choice([1, 2, 3])
        scratched_image = draw_random_walk_scratch(
            scratched_image, 
            length=length, 
            thickness=thickness
        )
        
    return scratched_image
