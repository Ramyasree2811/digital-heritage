import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def compute_ssim(img1, img2, channel_axis=2, data_range=None):
    """
    Computes Structural Similarity Index (SSIM) between two RGB/grayscale images.
    img1, img2: numpy arrays in range [0, 255] or [0, 1]
    """
    # Ensure they have same shape
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    if data_range is None:
        # Determine standard dynamic range according to image values
        max_val = max(float(img1.max()), float(img2.max()))
        data_range = 255.0 if max_val > 1.5 else 1.0

    # Check if grayscale or color
    if len(img1.shape) == 2:
        return float(compare_ssim(img1, img2, data_range=data_range))
    else:
        # For color images, specify the channel axis (typically axis 2)
        return float(compare_ssim(img1, img2, channel_axis=channel_axis, data_range=data_range))

if __name__ == "__main__":
    dummy1 = np.ones((256, 256, 3), dtype=np.uint8) * 200
    dummy2 = np.ones((256, 256, 3), dtype=np.uint8) * 190
    print("SSIM:", compute_ssim(dummy1, dummy2))
