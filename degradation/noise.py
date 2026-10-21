import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds Gaussian noise to an RGB image.
    image: numpy array (H, W, 3) in range [0, 255]
    """
    image = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    """
    Adds Salt & Pepper noise to an RGB image.
    image: numpy array (H, W, 3) in range [0, 255]
    """
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

def add_speckle_noise(image, sigma=0.1):
    """
    Adds Speckle (multiplicative) noise to an RGB image.
    image: numpy array (H, W, 3) in range [0, 255]
    """
    image = image.astype(np.float32)
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_noise(image, noise_type="gaussian", severity=0.5):
    """
    Applies noise based on noise_type and severity (0.0 to 1.0).
    """
    if severity <= 0:
        return image
        
    if noise_type == "gaussian":
        sigma = 5 + severity * 45  # Sigma range [5, 50]
        return add_gaussian_noise(image, sigma=sigma)
    elif noise_type == "salt_and_pepper":
        amount = 0.005 + severity * 0.045  # Amount range [0.5%, 5%]
        return add_salt_and_pepper_noise(image, amount=amount)
    elif noise_type == "speckle":
        sigma = 0.05 + severity * 0.25  # Sigma range [0.05, 0.30]
        return add_speckle_noise(image, sigma=sigma)
    return image
