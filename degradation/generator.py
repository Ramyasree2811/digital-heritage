import os
import random
import cv2
import numpy as np

# Import all individual degradation modules
from degradation.noise import apply_noise
from degradation.blur import apply_blur
from degradation.scratches import apply_scratches
from degradation.stains import apply_stains
from degradation.physical_defects import apply_physical_defects
from degradation.folds_wrinkles import apply_folds
from degradation.aging_lighting import apply_aging_lighting

class DegradationGenerator:
    """
    Coordinates and applies multiple degradations to clean images.
    Used for online data augmentation during model training and preprocessing.
    """
    def __init__(self, 
                 noise_prob=0.5, 
                 blur_prob=0.5, 
                 scratches_prob=0.5, 
                 stains_prob=0.5, 
                 physical_prob=0.4, 
                 folds_prob=0.4, 
                 aging_prob=0.6):
        self.probs = {
            "noise": noise_prob,
            "blur": blur_prob,
            "scratches": scratches_prob,
            "stains": stains_prob,
            "physical": physical_prob,
            "folds": folds_prob,
            "aging": aging_prob
        }

    def degrade_image(self, image, severity=None):
        """
        Applies a random combination of defects to an RGB image.
        image: numpy array (H, W, 3) in range [0, 255]
        severity: float in range [0.0, 1.0]. If None, a random severity is chosen.
        """
        if severity is None:
            severity = random.uniform(0.1, 0.9)
            
        degraded = image.copy()
        
        # 1. Physical defects (Tears, wormholes - applied first as they modify boundaries/geometry)
        if random.random() < self.probs["physical"]:
            p_sev = severity * random.uniform(0.7, 1.2)
            degraded = apply_physical_defects(degraded, severity=np.clip(p_sev, 0.0, 1.0))
            
        # 2. Folds and wrinkles
        if random.random() < self.probs["folds"]:
            f_sev = severity * random.uniform(0.6, 1.1)
            degraded = apply_folds(degraded, severity=np.clip(f_sev, 0.0, 1.0))
            
        # 3. Aging and uneven lighting (yellowing, contrast fade, shadow gradient)
        if random.random() < self.probs["aging"]:
            a_sev = severity * random.uniform(0.7, 1.3)
            degraded = apply_aging_lighting(degraded, severity=np.clip(a_sev, 0.0, 1.0))
            
        # 4. Water/ink stains and mold
        if random.random() < self.probs["stains"]:
            s_sev = severity * random.uniform(0.5, 1.2)
            degraded = apply_stains(degraded, severity=np.clip(s_sev, 0.0, 1.0))
            
        # 5. Scratches and scrapes
        if random.random() < self.probs["scratches"]:
            sc_sev = severity * random.uniform(0.5, 1.1)
            degraded = apply_scratches(degraded, severity=np.clip(sc_sev, 0.0, 1.0))
            
        # 6. Blurs (optical defocus/motion)
        if random.random() < self.probs["blur"]:
            b_sev = severity * random.uniform(0.4, 1.0)
            blur_types = ["gaussian", "motion", "defocus"]
            b_type = random.choice(blur_types)
            degraded = apply_blur(degraded, blur_type=b_type, severity=np.clip(b_sev, 0.0, 1.0))
            
        # 7. Noise (sensor noise / salt & pepper)
        if random.random() < self.probs["noise"]:
            n_sev = severity * random.uniform(0.4, 1.0)
            noise_types = ["gaussian", "salt_and_pepper", "speckle"]
            n_type = random.choice(noise_types)
            degraded = apply_noise(degraded, noise_type=n_type, severity=np.clip(n_sev, 0.0, 1.0))
            
        return degraded

# Test script interface
if __name__ == "__main__":
    # Test degradation on a dummy image
    dummy = np.ones((256, 256, 3), dtype=np.uint8) * 220
    # Draw some text
    cv2.putText(dummy, "Sanskrit Test Shloka", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    
    generator = DegradationGenerator()
    degraded = generator.degrade_image(dummy, severity=0.7)
    
    print("Degradation generator loaded successfully! Degraded image shape:", degraded.shape)
