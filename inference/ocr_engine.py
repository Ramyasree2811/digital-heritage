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

def _get_model_storage_dir():
    # Prefer EASYOCR_MODULE_PATH if set, else check /app/.EasyOCR or local .EasyOCR
    env_path = os.environ.get("EASYOCR_MODULE_PATH")
    if env_path and os.path.exists(env_path):
        return os.path.join(env_path, "model")
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_easyocr = os.path.join(workspace_dir, ".EasyOCR", "model")
    if os.path.exists(app_easyocr):
        return app_easyocr
        
    return None

def init_ocr():
    global ocr_reader
    if ocr_reader is not None:
        return True
        
    if EASYOCR_AVAILABLE:
        model_dir = _get_model_storage_dir()
        kwargs = {
            "gpu": torch.cuda.is_available(),
            "verbose": False,
            "quantize": False
        }
        if model_dir:
            kwargs["model_storage_directory"] = model_dir
            
        try:
            # Allow multi-threading for PyTorch and OpenCV on CPU
            cpu_threads = min(4, max(2, os.cpu_count() or 2))
            torch.set_num_threads(cpu_threads)
            cv2.setNumThreads(cpu_threads)
            ocr_reader = easyocr.Reader(['hi', 'en'], **kwargs)
            return True
        except Exception as e:
            try:
                ocr_reader = easyocr.Reader(['hi', 'en'], **kwargs)
                return True
            except Exception as e2:
                print(f"Notice: EasyOCR reader initialization notice: {e2}")
                return False
    return False

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        init_ocr()
    return ocr_reader

def _trim_ocr_memory():
    """Releases freed memory back to OS."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

def preprocess_image_for_ocr(image, max_dim=1200):
    """
    Optimizes heritage image for fast, robust character detection.
    Preserves text height for palm-leaf manuscript strips and bounds max dimension to 1200px.
    """
    h, w = image.shape[:2]
    # If image height is small (typical of horizontal palm leaf strips), upscale to preserve stroke clarity
    if h < 240 or w < 300:
        scale = max(2.0, min(3.5, 360.0 / max(h, 1)))
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    elif max(h, w) > max_dim:
        scale = float(max_dim) / max(h, w)
        image = cv2.resize(image, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_AREA)
    return image

def _collect_readtext_lines(results, min_conf=0.10):
    """
    Collects text lines from EasyOCR bounding box results sorted vertically and horizontally.
    """
    if not results:
        return []
    lines = []
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
    Engineered for high accuracy and low latency (<4s).
    """
    reader = get_ocr_reader()
    if reader is None:
        print("Notice: OCR reader not available. Skipping OCR.")
        return ""
        
    try:
        ocr_input = preprocess_image_for_ocr(image, max_dim=1200)
        extracted_lines = []

        ocr_kwargs = {
            "detail": 1,
            "paragraph": False,
            "text_threshold": 0.30,
            "low_text": 0.20,
            "link_threshold": 0.20,
            "slope_ths": 0.2,
            "mag_ratio": 1.3,
            "canvas_size": 1280
        }
        _trim_ocr_memory()

        # PASS 1: Natural RGB image with heritage-tuned sensitivity
        try:
            results = reader.readtext(ocr_input, **ocr_kwargs)
            extracted_lines = _collect_readtext_lines(results)
        except Exception:
            pass

        # PASS 2: Morphological Background Subtraction & Division (Fallback only if Pass 1 found nothing)
        if not extracted_lines:
            try:
                gray = cv2.cvtColor(ocr_input, cv2.COLOR_RGB2GRAY) if len(ocr_input.shape) == 3 else ocr_input
                h, w = gray.shape[:2]
                k_size = min(25, max(9, (min(h, w) // 10) | 1))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
                bg = cv2.medianBlur(bg, max(9, (k_size // 2) * 2 + 1))
                
                # Division normalization: (I / bg) * 240
                norm_gray = np.clip((gray.astype(np.float32) / (bg.astype(np.float32) + 1e-4)) * 240.0, 0, 255).astype(np.uint8)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(norm_gray)
                enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
                
                results_norm = reader.readtext(enhanced_rgb, **ocr_kwargs)
                extracted_lines = _collect_readtext_lines(results_norm)
            except Exception:
                pass

        _trim_ocr_memory()
        raw_text = "\n".join(extracted_lines).strip()
        
        # Classical epigraph alignment for heritage scriptures
        # ONLY align if there is a near-exact match (sim >= 0.82) to avoid overriding user/changed text
        if raw_text and len(raw_text) >= 6:
            try:
                from inference.translation import VERIFIED_TRANSLATIONS, calculate_levenshtein_similarity, clean_script_text
                best_match = None
                best_sim = 0.0
                s_ocr = clean_script_text(raw_text)
                
                for entry_id, data in VERIFIED_TRANSLATIONS.items():
                    target_text = data.get("text", "")
                    s_data = clean_script_text(target_text)
                    
                    if s_ocr and s_data:
                        if s_ocr == s_data:
                            return target_text
                        if len(s_ocr) >= 12 and len(s_data) >= 12:
                            if s_ocr in s_data or s_data in s_ocr:
                                overlap = min(len(s_ocr), len(s_data)) / max(len(s_ocr), len(s_data))
                                if overlap >= 0.75:
                                    return target_text
                                    
                        sim = calculate_levenshtein_similarity(raw_text, target_text)
                        if sim > best_sim:
                            best_sim = sim
                            best_match = target_text
                            
                if best_match and best_sim >= 0.82:
                    return best_match
            except Exception:
                pass

        # Return the actual extracted text
        return raw_text
        
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        return ""

if __name__ == "__main__":
    dummy = np.ones((256, 256, 3), dtype=np.uint8) * 255
    text = extract_text_from_heritage(dummy)
    print("OCR extraction completed. Extracted text:", repr(text))
