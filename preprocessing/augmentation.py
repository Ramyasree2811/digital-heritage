import albumentations as A
import numpy as np

def get_monument_augmentation():
    """
    Augmentation pipeline for natural photos/paintings/monuments.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15, p=0.5, border_mode=0, cval=(0,0,0)),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.Perspective(scale=(0.02, 0.05), p=0.3, cval=(0,0,0))
    ])

def get_document_augmentation():
    """
    Augmentation pipeline for historical text documents.
    """
    return A.Compose([
        # Text shouldn't be flipped vertically or horizontally as it changes characters,
        # but small rotations, perspective shifts, and lighting variations are highly realistic.
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5, border_mode=0, cval=(255,255,255)),
        A.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, p=0.4),
        A.Perspective(scale=(0.01, 0.04), p=0.3, cval=(255,255,255))
    ])

def get_palm_leaf_augmentation():
    """
    Augmentation pipeline for palm-leaf manuscripts.
    """
    return A.Compose([
        # Small rotations and perspective shifts represent hand-held scanner variance
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=6, p=0.5, border_mode=0, cval=(210, 180, 130)),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, p=0.4),
        A.Perspective(scale=(0.02, 0.04), p=0.3, cval=(10,10,10)) # shelf background is usually black
    ])

def augment_image(image, domain="monuments"):
    """
    Applies albumentations to a clean image based on its domain.
    """
    if domain == "monuments":
        aug = get_monument_augmentation()
    elif domain == "documents":
        aug = get_document_augmentation()
    else:
        aug = get_palm_leaf_augmentation()
        
    augmented = aug(image=image)
    return augmented['image']
