import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure workspace directory is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

import time
import json
import argparse
import cv2
import numpy as np

# Import pipeline steps
from inference.restore import restore_heritage_image
from inference.ocr_engine import extract_text_from_heritage
from inference.translation import translate_heritage_text
from inference.super_resolution import upscale_image

OUT_DIR = os.path.join(WORKSPACE_DIR, "outputs", "pipeline")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "results")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def detect_image_damage(image, domain="palm_leaf"):
    """
    Identifies actual physical defects (wormholes, tears, paper cracks, fungal stains)
    while strictly protecting inscribed text characters.
    """
    h, w, c = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. Detect extreme border voids / tears (pure white / pure black scan margins)
    border_loss = ((gray > 248) | (gray < 8)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    border_loss = cv2.morphologyEx(border_loss, cv2.MORPH_OPEN, kernel)
    mask = cv2.bitwise_or(mask, border_loss)
    
    # 2. Protect text strokes from damage marking
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    text_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    text_protect = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    # 3. Detect large damp stains / blotches outside text strokes
    if domain in ["palm_leaf", "documents"]:
        bg_blur = cv2.medianBlur(gray, 31)
        stains = cv2.adaptiveThreshold(bg_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15)
        stains_clean = cv2.morphologyEx(stains, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = cv2.bitwise_or(mask, stains_clean)
        
    # Subtract protected text strokes so inpainting/restoration never erases text!
    mask[text_protect > 0] = 0
    return mask

def run_heritage_pipeline(image_path, model_name="best", domain="palm_leaf", target_lang="en", run_ocr=True, run_sr=True, status_callback=None):
    """
    Orchestrates the entire Digital Heritage Restoration Pipeline.
    Loads trained models without triggering retraining.
    """
    t0 = time.time()
    def log_status(step, progress):
        if status_callback:
            status_callback(step, progress)
        print(f"[{progress}%] {step}")
        
    # Load Image
    log_status("Preprocessing Image", 10)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 1: Detect Damage
    log_status("Detecting Damage", 25)
    damage_mask = detect_image_damage(image_rgb, domain=domain)
    
    # Step 2: Restore Image
    log_status(f"Restoring Image (Using Model: {model_name})", 45)
    is_doc = (domain != "monuments")
    restored = restore_heritage_image(image_rgb, model_name=model_name, damage_mask=damage_mask, is_document=is_doc)
    
    # Step 3: Run OCR
    extracted_text = ""
    if run_ocr:
        log_status("Running OCR (Extracting Text)", 65)
        extracted_text = extract_text_from_heritage(restored)
        if not extracted_text:
            extracted_text = "No inscribed text detected in this scan region."
    else:
        log_status("Skipping OCR", 65)
        extracted_text = "OCR disabled"
        
    # Step 4: Run Translation
    translated_text = ""
    if run_ocr and extracted_text and not extracted_text.startswith("No inscribed") and not extracted_text.startswith("OCR disabled"):
        log_status("Translating Text", 80)
        translated_text = translate_heritage_text(extracted_text, target_lang=target_lang)
    else:
        translated_text = "N/A"
        
    # Step 5: Super-Resolution Upscaling (2x)
    restored_sr = restored
    if run_sr:
        log_status("Enhancing Image (Running Super-Resolution)", 90)
        restored_sr = upscale_image(restored, scale_factor=2)
        
    # Save standard pipeline cache outputs
    filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(filename)
    
    mask_out_path = os.path.join(OUT_DIR, f"{base_name}_damage_mask.png")
    restored_out_path = os.path.join(OUT_DIR, f"{base_name}_restored.png")
    sr_out_path = os.path.join(OUT_DIR, f"{base_name}_high_res.png")
    
    cv2.imwrite(mask_out_path, damage_mask)
    cv2.imwrite(restored_out_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(sr_out_path, cv2.cvtColor(restored_sr, cv2.COLOR_RGB2BGR))
    
    total_time = round(time.time() - t0, 3)
    log_status("Pipeline Execution Completed!", 100)
    
    return {
        "original_path": image_path,
        "mask_path": mask_out_path,
        "restored_path": restored_out_path,
        "high_res_path": sr_out_path,
        "extracted_text": extracted_text,
        "translation": translated_text,
        "processing_time": total_time,
        "model_used": model_name
    }

def run_cli_inference(image_path, model_name="best", domain="palm_leaf", run_ocr=True, run_sr=True):
    """
    Dedicated CLI Inference Runner:
    Creates a unique timestamped results folder in outputs/results/<timestamp>/
    """
    timestamp = time.strftime("%Y_%m_%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 70)
    print(f" Running Heritage Inference on: {image_path}")
    print(f" Target Result Directory: {run_dir}")
    print("=" * 70)
    
    results = run_heritage_pipeline(
        image_path=image_path,
        model_name=model_name,
        domain=domain,
        run_ocr=run_ocr,
        run_sr=run_sr
    )
    
    # Copy/Save into outputs/results/<timestamp>/
    dst_orig = os.path.join(run_dir, "original.png")
    dst_restored = os.path.join(run_dir, "restored.png")
    dst_enhanced = os.path.join(run_dir, "enhanced.png")
    dst_mask = os.path.join(run_dir, "damage_mask.png")
    dst_text = os.path.join(run_dir, "extracted_text.txt")
    dst_trans = os.path.join(run_dir, "translated_text.txt")
    dst_json = os.path.join(run_dir, "result.json")
    
    import shutil
    shutil.copy2(image_path, dst_orig)
    shutil.copy2(results["restored_path"], dst_restored)
    shutil.copy2(results["high_res_path"], dst_enhanced)
    shutil.copy2(results["mask_path"], dst_mask)
    
    with open(dst_text, "w", encoding="utf-8") as f:
        f.write(results["extracted_text"])
        
    with open(dst_trans, "w", encoding="utf-8") as f:
        f.write(results["translation"])
        
    result_metadata = {
        "input_filename": os.path.basename(image_path),
        "model_used": results["model_used"],
        "timestamp": timestamp,
        "processing_time_seconds": results["processing_time"],
        "domain": domain,
        "extracted_text": results["extracted_text"],
        "translation": results["translation"],
        "files": {
            "original": dst_orig,
            "restored": dst_restored,
            "enhanced": dst_enhanced,
            "mask": dst_mask
        }
    }
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(result_metadata, f, indent=4)
        
    print("\n" + "=" * 70)
    print(f" SUCCESS: Results saved to: {run_dir}")
    print(f" Processing Time: {results['processing_time']}s")
    if results["extracted_text"]:
        print(f" Extracted Text: {results['extracted_text'][:100]}...")
    print("=" * 70)
    return run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Heritage Restoration Inference Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input heritage image")
    parser.add_argument("--model", type=str, default="best", help="Model name (default: 'best' loads outputs/best_model/model.pth)")
    parser.add_argument("--domain", type=str, default="palm_leaf", choices=["palm_leaf", "documents", "monuments"], help="Heritage domain")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR text extraction")
    parser.add_argument("--no-sr", action="store_true", help="Disable Super-Resolution enhancement")
    
    args = parser.parse_args()
    run_cli_inference(
        image_path=args.input,
        model_name=args.model,
        domain=args.domain,
        run_ocr=not args.no_ocr,
        run_sr=not args.no_sr
    )
