import cv2
import numpy as np

def resize_and_pad(image, target_size=256, keep_aspect=True):
    """
    Resizes an RGB/grayscale image to target_size x target_size.
    If keep_aspect is True, preserves aspect ratio and pads the rest with white (255) for documents
    or black (0) for natural images.
    """
    h, w = image.shape[:2]
    
    if not keep_aspect:
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
    # Calculate scale factor
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas (default to white for padding, representing clean paper)
    # Check if image is grayscale or color
    if len(image.shape) == 3:
        # Determine if it's text/paper (high average value) or photo
        mean_val = np.mean(image)
        pad_color = (255, 255, 255) if mean_val > 127 else (0, 0, 0)
        canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
    else:
        pad_color = 255 if np.mean(image) > 127 else 0
        canvas = np.ones((target_size, target_size), dtype=np.uint8) * pad_color
        
    # Place resized image on canvas (center it)
    dy = (target_size - new_h) // 2
    dx = (target_size - new_w) // 2
    
    if len(image.shape) == 3:
        canvas[dy:dy+new_h, dx:dx+new_w, :] = resized
    else:
        canvas[dy:dy+new_h, dx:dx+new_w] = resized
        
    return canvas

if __name__ == "__main__":
    # Test resizing
    dummy = np.zeros((100, 200, 3), dtype=np.uint8)
    out = resize_and_pad(dummy, target_size=256)
    print("Resize utility test successful! Output shape:", out.shape)
