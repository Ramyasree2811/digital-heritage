import os
import cv2
import numpy as np
import random
import shutil
import pandas as pd
from PIL import Image

# Import preprocessing utilities
from preprocessing.resize import resize_and_pad
from preprocessing.degradation import apply_general_degradation
from preprocessing.palm_leaf_degradation import apply_palm_leaf_degradation
from preprocessing.augmentation import augment_image

# Directories
WORKSPACE_DIR = r"c:\Users\DELL\OneDrive\Desktop\digital"
DATASETS_DIR = r"C:\Users\DELL\OneDrive\Desktop\Digital-Heritage-Restoration\datasets"
DATASET_OUT = os.path.join(WORKSPACE_DIR, "dataset")

# Target domains
DOMAINS = ["monuments", "documents", "palm_leaf"]

# Setup directories
for dom in DOMAINS:
    for folder in ["original", "degraded", "masks"]:
        os.makedirs(os.path.join(DATASET_OUT, dom, folder), exist_ok=True)
        
for split in ["train", "validation", "test"]:
    for sub in ["clean", "degraded", "masks"]:
        os.makedirs(os.path.join(DATASET_OUT, split, sub), exist_ok=True)

def collect_source_files():
    """
    Collects clean heritage images from the various folders in datasets/
    """
    sources = {
        "monuments": [],
        "documents": [],
        "palm_leaf": []
    }
    
    # 1. Monuments source files
    paintings_dir = os.path.join(DATASETS_DIR, "source_paintings")
    if os.path.exists(paintings_dir):
        sources["monuments"].extend([os.path.join(paintings_dir, f) for f in os.listdir(paintings_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp'))])
        
    artifacts_dir = os.path.join(DATASETS_DIR, "source_artifacts")
    if os.path.exists(artifacts_dir):
        sources["monuments"].extend([os.path.join(artifacts_dir, f) for f in os.listdir(artifacts_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp'))])
        
    # 2. Documents source files
    docs_dir = os.path.join(DATASETS_DIR, "historical_documents_edge")
    if os.path.exists(docs_dir):
        sources["documents"].extend([os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp'))])
        
    inscr_dir = os.path.join(DATASETS_DIR, "source_inscriptions")
    if os.path.exists(inscr_dir):
        sources["documents"].extend([os.path.join(inscr_dir, f) for f in os.listdir(inscr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp'))])
        
    # Sample 100 clean images from synthetic/clean (which are mostly text)
    synthetic_clean_dir = os.path.join(DATASETS_DIR, "synthetic", "clean")
    if os.path.exists(synthetic_clean_dir):
        syn_files = [os.path.join(synthetic_clean_dir, f) for f in os.listdir(synthetic_clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp'))]
        if syn_files:
            random.seed(42)
            sources["documents"].extend(random.sample(syn_files, min(len(syn_files), 100)))
            
    # 3. Palm Leaf source files
    manuscript_dir = os.path.join(DATASETS_DIR, "ancient_library_manuscripts")
    if os.path.exists(manuscript_dir):
        sources["palm_leaf"].extend([os.path.join(manuscript_dir, f) for f in os.listdir(manuscript_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp'))])
        
    vedocr_dir = os.path.join(DATASETS_DIR, "vedocr", "line_crops")
    if os.path.exists(vedocr_dir):
        sources["palm_leaf"].extend([os.path.join(vedocr_dir, f) for f in os.listdir(vedocr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp'))])
        
    print("Collected Source files count:")
    for k, v in sources.items():
        print(f"- {k}: {len(v)} files")
        
    return sources

def build_domain_dataset(sources, target_count=1000):
    """
    Creates target_count pairs for each domain using augmentation and degradation.
    """
    metadata = []
    
    for domain in DOMAINS:
        print(f"\nProcessing domain: {domain}...")
        files = sources[domain]
        if not files:
            print(f"Error: No source files for domain {domain}. Skipping.")
            continue
            
        count = 0
        random.seed(42)
        
        while count < target_count:
            # Randomly select a source file
            src_file = random.choice(files)
            
            # Read image
            image = cv2.imread(src_file)
            if image is None:
                continue
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. Resize and Pad (Standardize to 256x256)
            # Palm leaves look better with black background pad, documents with white pad
            keep_aspect = True
            image_resized = resize_and_pad(image, target_size=256, keep_aspect=keep_aspect)
            
            # 2. Augmentation (Geometric & Color Shift to create new variations)
            image_aug = augment_image(image_resized, domain=domain)
            
            # 3. Apply Domain-Specific Degradation and Get Damage Mask
            severity = random.uniform(0.2, 0.8)
            if domain == "palm_leaf":
                image_deg, damage_mask = apply_palm_leaf_degradation(image_aug, severity=severity)
                damage_type = "palm_leaf_decay+cracks+stains"
            else:
                image_deg, damage_mask = apply_general_degradation(image_aug, severity=severity)
                damage_type = "noise+blur+scratches+stains"
                
            # Filenames
            filename = f"{domain}_{count:04d}.png"
            
            # Save original (clean), degraded, and masks in the domain folder
            original_path = os.path.join(DATASET_OUT, domain, "original", filename)
            degraded_path = os.path.join(DATASET_OUT, domain, "degraded", filename)
            mask_path = os.path.join(DATASET_OUT, domain, "masks", filename)
            
            # Save using OpenCV (convert back to BGR for saving)
            cv2.imwrite(original_path, cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR))
            cv2.imwrite(degraded_path, cv2.cvtColor(image_deg, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, damage_mask)  # Grayscale mask saves directly
            
            metadata.append({
                "filename": filename,
                "domain": domain,
                "damage_type": damage_type,
                "severity": round(severity, 2)
            })
            
            count += 1
            if count % 200 == 0:
                print(f"Generated {count}/{target_count} images for {domain}.")
                
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(DATASET_OUT, "metadata.csv"), index=False)
    print("\nAll domains processed! Metadata saved.")
    return df

def split_dataset(metadata_df, train_pct=0.7, val_pct=0.15):
    """
    Splits the generated dataset into train, validation, and test folders.
    """
    print("\nSplitting dataset into train, validation, and test splits...")
    
    train_metadata = []
    val_metadata = []
    test_metadata = []
    
    # Split separately for each domain to ensure balanced splits
    for domain in DOMAINS:
        dom_df = metadata_df[metadata_df["domain"] == domain].copy()
        dom_df = dom_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_total = len(dom_df)
        n_train = int(n_total * train_pct)
        n_val = int(n_total * val_pct)
        
        train_slice = dom_df.iloc[:n_train]
        val_slice = dom_df.iloc[n_train:n_train+n_val]
        test_slice = dom_df.iloc[n_train+n_val:]
        
        def copy_files(slice_df, split_name, metadata_list):
            for idx, row in slice_df.iterrows():
                fname = row["filename"]
                dom = row["domain"]
                
                # Src paths
                src_orig = os.path.join(DATASET_OUT, dom, "original", fname)
                src_deg = os.path.join(DATASET_OUT, dom, "degraded", fname)
                src_mask = os.path.join(DATASET_OUT, dom, "masks", fname)
                
                # Dst paths
                dst_orig = os.path.join(DATASET_OUT, split_name, "clean", fname)
                dst_deg = os.path.join(DATASET_OUT, split_name, "degraded", fname)
                dst_mask = os.path.join(DATASET_OUT, split_name, "masks", fname)
                
                shutil.copy2(src_orig, dst_orig)
                shutil.copy2(src_deg, dst_deg)
                shutil.copy2(src_mask, dst_mask)
                
                metadata_list.append(row.to_dict())
                
        copy_files(train_slice, "train", train_metadata)
        copy_files(val_slice, "validation", val_metadata)
        copy_files(test_slice, "test", test_metadata)
        
        print(f"Split {domain}: {len(train_slice)} train, {len(val_slice)} val, {len(test_slice)} test.")
        
    # Save split metadata CSVs
    pd.DataFrame(train_metadata).to_csv(os.path.join(DATASET_OUT, "train_metadata.csv"), index=False)
    pd.DataFrame(val_metadata).to_csv(os.path.join(DATASET_OUT, "val_metadata.csv"), index=False)
    pd.DataFrame(test_metadata).to_csv(os.path.join(DATASET_OUT, "test_metadata.csv"), index=False)
    
    print("\nDataset split successfully created under dataset/train, dataset/validation, and dataset/test!")

def main():
    print("Starting Multi-Domain Digital Heritage Dataset Generation...")
    sources = collect_source_files()
    metadata_df = build_domain_dataset(sources, target_count=1000)
    split_dataset(metadata_df)
    print("\nDataset generation completed successfully!")

if __name__ == "__main__":
    main()
