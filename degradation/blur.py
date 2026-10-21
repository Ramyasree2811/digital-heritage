import cv2
import numpy as np

def apply_gaussian_blur(image, kernel_size=5):
    """
    Applies Gaussian blur to an RGB image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_motion_blur(image, size=15, angle=45):
    """
    Applies motion blur to an RGB image.
    """
    # Create the motion blur kernel
    kernel = np.zeros((size, size))
    # Fill kernel with a line of ones representing the motion direction
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    
    center = size // 2
    for i in range(size):
        offset = i - center
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
            
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)

def apply_defocus_blur(image, radius=5):
    """
    Applies defocus (disk) blur to an RGB image.
    """
    # Create circular kernel
    size = radius * 2 + 1
    kernel = np.zeros((size, size))
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)

def apply_blur(image, blur_type="gaussian", severity=0.5):
    """
    Applies blur based on blur_type and severity (0.0 to 1.0).
    """
    if severity <= 0:
        return image
        
    if blur_type == "gaussian":
        ksize = int(3 + severity * 12)  # Kernel size [3, 15]
        if ksize % 2 == 0:
            ksize += 1
        return apply_gaussian_blur(image, kernel_size=ksize)
    elif blur_type == "motion":
        size = int(5 + severity * 20)  # Motion kernel size [5, 25]
        angle = np.random.randint(0, 360)
        return apply_motion_blur(image, size=size, angle=angle)
    elif blur_type == "defocus":
        radius = int(2 + severity * 8)  # Defocus radius [2, 10]
        return apply_defocus_blur(image, radius=radius)
    return image
