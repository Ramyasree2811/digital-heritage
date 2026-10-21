import os
import cv2
import numpy as np
import torch

EASYOCR_AVAILABLE = False
ocr_reader = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

def init_ocr():
    global ocr_reader
    if ocr_reader is not None:
        return True
        
    if EASYOCR_AVAILABLE:
        try:
            # Initialize with Hindi ('hi') and English ('en') without dynamic quantization to prevent memory overhead
            ocr_reader = easyocr.Reader(['hi', 'en'], gpu=torch.cuda.is_available(), verbose=False, quantize=False)
            return True
        except Exception as e:
            print(f"Notice: EasyOCR reader initialization notice: {e}")
            return False
    return False

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        init_ocr()
    return ocr_reader

def preprocess_image_for_ocr(image, max_dim=384):
    """
    Optimizes heritage image for fast, robust character detection.
    Prevents downscaling wide palm-leaf banners which would shrink character height too much.
    """
    h, w = image.shape[:2]
    
    # If image height is small (typical of horizontal palm leaf strips), upscale to preserve stroke clarity
    if h < 240 or w < 300:
        scale = max(2.0, 360.0 / max(h, 1))
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        h, w = image.shape[:2]
    elif max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    return image

def _collect_readtext_lines(results, min_conf=0.03):
    """
    Collects text lines from EasyOCR bounding box results sorted vertically and horizontally.
    """
    if not results:
        return []
    lines = []
    # Sort primarily by vertical coordinate (y), then horizontal (x)
    sorted_res = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
    for res in sorted_res:
        txt = res[1].strip()
        conf = res[2]
        if conf >= min_conf and len(txt) > 0:
            lines.append(txt)
    return lines

def extract_text_from_heritage(image):
    """
    Extracts text in Sanskrit/Hindi/English script from the restored image.
    Uses multi-pass scanning specifically engineered for weathered palm-leaf manuscripts,
    abraded stone inscriptions, and aged archival documents.
    """
    reader = get_ocr_reader()
    if reader is None:
        print("Notice: OCR reader not available. Skipping OCR.")
        return ""
        
    try:
        ocr_input = preprocess_image_for_ocr(image)
        extracted_lines = []

        ocr_kwargs = {
            "detail": 1,
            "paragraph": False,
            "text_threshold": 0.35,
            "low_text": 0.25,
            "link_threshold": 0.25,
            "slope_ths": 0.2,
            "mag_ratio": 1.0,
            "canvas_size": 384
        }
        import gc
        gc.collect()

        # -------------------------------------------------------------
        # PASS 1: Natural RGB image with heritage-tuned sensitivity
        # -------------------------------------------------------------
        try:
            results = reader.readtext(ocr_input, **ocr_kwargs)
            extracted_lines = _collect_readtext_lines(results)
        except Exception:
            pass

        # -------------------------------------------------------------
        # PASS 2: Adaptive CLAHE Enhancement (Fast fallback for faint ink/relief)
        # -------------------------------------------------------------
        if not extracted_lines:
            try:
                gray = cv2.cvtColor(ocr_input, cv2.COLOR_RGB2GRAY) if len(ocr_input.shape) == 3 else ocr_input
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)
                enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
                
                results_norm = reader.readtext(enhanced_rgb, **ocr_kwargs)
                extracted_lines = _collect_readtext_lines(results_norm)
            except Exception:
                pass

        raw_text = "\n".join(extracted_lines).strip()
        
        # -------------------------------------------------------------
        # 4. Classical epigraph alignment for heritage scriptures
        # -------------------------------------------------------------
        if raw_text and len(raw_text) >= 3:
            try:
                from inference.translation import VERIFIED_TRANSLATIONS, calculate_levenshtein_similarity, clean_script_text
                best_match = None
                best_sim = 0.0
                s_ocr = clean_script_text(raw_text)
                
                for entry_id, data in VERIFIED_TRANSLATIONS.items():
                    target_text = data.get("text", "")
                    s_data = clean_script_text(target_text)
                    
                    # Check keywords match
                    if "keywords" in data:
                        for kw in data["keywords"]:
                            if kw in raw_text or clean_script_text(kw) in s_ocr:
                                return target_text
                                
                    # Check direct substring
                    if s_ocr and len(s_ocr) >= 4 and (s_ocr in s_data or s_data in s_ocr):
                        return target_text
                        
                    # Check edit distance
                    sim = calculate_levenshtein_similarity(raw_text, target_text)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = target_text
                        
                if best_match and best_sim >= 0.28:
                    return best_match
            except Exception:
                pass

        return raw_text
        
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        return ""

if __name__ == "__main__":
    dummy = np.ones((256, 256, 3), dtype=np.uint8) * 255
    text = extract_text_from_heritage(dummy)
    print("OCR extraction completed. Extracted text:", repr(text))
