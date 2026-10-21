"""
Unified Inference Pipeline for Digital Heritage Restoration
Loads auto-selected best model or user-selected neural architecture (Restormer, SwinIR, DnCNN, U-Net, Pix2Pix),
restores user-uploaded heritage images with domain-adaptive conditioning and Hann-window tiled blending,
computes honest quality metrics against ground truth, performs multi-pass OCR extraction,
and executes multi-language scholarly translation.
"""

import os
import sys
import uuid
import json
import time
import cv2
import numpy as np
import torch

# Ensure UTF-8 output on Windows shells to prevent UnicodeEncodeError on Indic/Sanskrit text
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from models.unet.model import UNet
from models.dncnn.model import DnCNN
from models.pix2pix.model import Pix2PixGenerator
from models.restormer.model import Restormer
from models.swinir.model import SwinIR

from training.utils import compute_psnr, get_device
from evaluation.ssim import compute_ssim
import importlib.util

def _load_helper_module(mod_name, file_rel_path):
    fpath = os.path.join(WORKSPACE_DIR, file_rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ocr_mod = _load_helper_module("heritage_ocr_engine", os.path.join("inference", "ocr_engine.py"))
extract_text_from_heritage = _ocr_mod.extract_text_from_heritage

_trans_mod = _load_helper_module("heritage_trans_engine", os.path.join("inference", "translation.py"))
translate_heritage_text = _trans_mod.translate_heritage_text

MODELS_BEST_DIR = os.path.join(WORKSPACE_DIR, "models", "best_model")
OUTPUTS_BEST_DIR = os.path.join(WORKSPACE_DIR, "outputs", "best_model")
CHECKPOINTS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "checkpoints")
UPLOADS_DIR = os.path.join(WORKSPACE_DIR, "static", "uploads")
OUTPUTS_DIR = os.path.join(WORKSPACE_DIR, "static", "outputs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

class HeritageRestorer:
    """
    Research-Grade Digital Heritage Restoration and Transcription Engine.
    Equipped with multi-model caching, native resolution tiling, and domain-adaptive enhancement.
    """
    def __init__(self):
        self.device = get_device()
        self._model_cache = {}
        self.best_model_name, self.config = self._read_best_config()
        # Preload the best model into cache
        self.get_model("best")

    def _read_best_config(self):
        cfg_path = os.path.join(MODELS_BEST_DIR, "model_config.json")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(OUTPUTS_BEST_DIR, "model_config.json")
        config = {}
        model_name = "Restormer"
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    model_name = config.get("model_name", "Restormer")
            except Exception as e:
                print(f"[!] Warning reading best model config: {e}")
        return model_name, config

    def get_model(self, model_name="best"):
        """
        Dynamically retrieves or instantiates the requested architecture,
        loading trained checkpoints from outputs/checkpoints or models/best_model.
        Caches models in memory for instantaneous sub-millisecond switching.
        """
        req_clean = (model_name or "best").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        if req_clean in ["best", "auto", "bestmodel"]:
            req_clean = self.best_model_name.lower().replace("-", "").replace("_", "")

        # Map to canonical model key
        if "swin" in req_clean:
            key = "swinir"
            display_name = "SwinIR"
        elif "dncnn" in req_clean:
            key = "dncnn"
            display_name = "DnCNN"
        elif "pix" in req_clean:
            key = "pix2pix"
            display_name = "Pix2Pix"
        elif "unet" in req_clean:
            key = "unet"
            display_name = "U-Net"
        else:
            key = "restormer"
            display_name = "Restormer"

        if key in self._model_cache:
            return self._model_cache[key], display_name

        # Instantiate model architecture
        if key == "restormer":
            model = Restormer(in_channels=3, out_channels=3, num_blocks=[1, 1, 1], channels=32)
            ckpt_candidates = [
                os.path.join(MODELS_BEST_DIR, "model.pth"),
                os.path.join(CHECKPOINTS_DIR, "restormer", "best.pth"),
                os.path.join(OUTPUTS_BEST_DIR, "model.pth")
            ]
        elif key == "swinir":
            model = SwinIR(in_channels=3, out_channels=3, img_size=256, channels=32, num_heads=4)
            ckpt_candidates = [
                os.path.join(CHECKPOINTS_DIR, "swinir", "best.pth")
            ]
        elif key == "dncnn":
            model = DnCNN(in_channels=3, out_channels=3, num_layers=8, num_features=32)
            ckpt_candidates = [
                os.path.join(CHECKPOINTS_DIR, "dncnn", "best.pth")
            ]
        elif key == "pix2pix":
            model = Pix2PixGenerator(in_channels=3, out_channels=3)
            ckpt_candidates = [
                os.path.join(CHECKPOINTS_DIR, "pix2pix", "best_generator.pth"),
                os.path.join(CHECKPOINTS_DIR, "pix2pix", "best.pth")
            ]
        else: # unet
            model = UNet(in_channels=3, out_channels=3)
            ckpt_candidates = [
                os.path.join(CHECKPOINTS_DIR, "unet", "best.pth")
            ]

        # Load weights safely
        loaded = False
        for p in ckpt_candidates:
            if os.path.exists(p):
                try:
                    ckpt = torch.load(p, map_location=self.device)
                    state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
                    model.load_state_dict(state)
                    print(f"[+] Loaded {display_name} weights successfully from: {p}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[!] Warning loading {display_name} weights from {p}: {e}")

        if not loaded:
            print(f"[!] Notice: Checkpoint not found for {display_name}. Using initialized network.")

        model.to(self.device)
        model.eval()
        self._model_cache[key] = model
        return model, display_name

    def _auto_detect_domain(self, image_rgb):
        """
        Determines whether the input image is a palm leaf, paper document, or stone monument
        based on HSV saturation and color distribution.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        sat_mean = np.mean(hsv[:, :, 1])
        hue_mean = np.mean(hsv[:, :, 0])
        val_mean = np.mean(hsv[:, :, 2])

        # Very low saturation -> stone monument / rock inscription or grayscale scan
        if sat_mean < 30.0:
            return "monuments"
        # Golden brown warm hue (10-35) with moderate-to-high saturation -> palm leaf
        elif 10.0 <= hue_mean <= 38.0 and sat_mean >= 40.0:
            return "palm_leaf"
        else:
            return "documents"

    def _apply_domain_adaptive_enhancement(self, original_rgb, restored_raw, domain="palm_leaf"):
        """
        Applies domain-specific LAB bilateral background normalization and stroke sharpening
        engineered for historical substrates (palm leaves, cotton-rag manuscripts, carved stone).
        """
        h, w = restored_raw.shape[:2]
        lab = cv2.cvtColor(restored_raw, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        # Structuring element size adapted to image resolution
        max_dim = max(h, w)
        min_dim = min(h, w)
        max_bound = max(3, (min_dim - 1) | 1)

        if domain == "palm_leaf":
            # Longitudinal palm-leaf fibers: elliptical filter to cancel horizontal wood grain
            k_size = min(max_bound, max(5, (min_dim // 5) | 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            bg = cv2.morphologyEx(l_chan, cv2.MORPH_DILATE, kernel)
            blur_k = min(max_bound, max(3, (k_size // 2) * 2 + 1))
            bg = cv2.medianBlur(bg, blur_k)

            # Division normalization (levels uneven golden-brown leaf color)
            norm_l = np.clip((l_chan.astype(np.float32) / (bg.astype(np.float32) + 1e-4)) * 236.0, 0, 255).astype(np.uint8)
            denoised_l = cv2.bilateralFilter(norm_l, d=5, sigmaColor=35, sigmaSpace=35)

            # Recover incised character stroke density
            ink_mask = np.clip((215.0 - denoised_l.astype(np.float32)) / 75.0, 0.0, 1.0)
            enhanced_l = np.clip(denoised_l.astype(np.float32) - (ink_mask * 26.0), 0, 255).astype(np.uint8)

            # Gentle unsharp mask
            blurred = cv2.GaussianBlur(enhanced_l, (0, 0), 1.0)
            sharp_l = np.clip(cv2.addWeighted(enhanced_l, 1.25, blurred, -0.25, 0), 0, 255).astype(np.uint8)

            denoised_a = cv2.bilateralFilter(a_chan, d=5, sigmaColor=20, sigmaSpace=20)
            denoised_b = cv2.bilateralFilter(b_chan, d=5, sigmaColor=20, sigmaSpace=20)
            return cv2.cvtColor(cv2.merge((sharp_l, denoised_a, denoised_b)), cv2.COLOR_LAB2RGB)

        elif domain == "monuments":
            # Stone & Rock Epigraphy: preserve 3D chisel relief and lithic grain
            k_size = min(max_bound, max(3, (min_dim // 8) | 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            bg = cv2.morphologyEx(l_chan, cv2.MORPH_DILATE, kernel)
            blur_k = min(max_bound, max(3, (k_size // 2) * 2 + 1))
            bg = cv2.medianBlur(bg, blur_k)

            norm_l = np.clip((l_chan.astype(np.float32) / (bg.astype(np.float32) + 1e-4)) * 220.0, 0, 255).astype(np.uint8)
            # Edge-preserving filter for chiseled crevices
            denoised_l = cv2.bilateralFilter(norm_l, d=5, sigmaColor=25, sigmaSpace=25)

            # Local CLAHE contrast for faint engraved epigraphs
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_l = clahe.apply(denoised_l)

            denoised_a = cv2.bilateralFilter(a_chan, d=3, sigmaColor=15, sigmaSpace=15)
            denoised_b = cv2.bilateralFilter(b_chan, d=3, sigmaColor=15, sigmaSpace=15)
            return cv2.cvtColor(cv2.merge((clahe_l, denoised_a, denoised_b)), cv2.COLOR_LAB2RGB)

        else: # documents & manuscripts
            # Paper foxing, yellowing, and verso bleed-through attenuation
            k_size = min(max_bound, max(5, (min_dim // 6) | 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            bg = cv2.morphologyEx(l_chan, cv2.MORPH_DILATE, kernel)
            blur_k = min(max_bound, max(3, (k_size // 2) * 2 + 1))
            bg = cv2.medianBlur(bg, blur_k)

            norm_l = np.clip((l_chan.astype(np.float32) / (bg.astype(np.float32) + 1e-4)) * 242.0, 0, 255).astype(np.uint8)
            denoised_l = cv2.bilateralFilter(norm_l, d=5, sigmaColor=30, sigmaSpace=30)

            # Ink stroke sharpening
            blurred = cv2.GaussianBlur(denoised_l, (0, 0), 1.0)
            sharp_l = np.clip(cv2.addWeighted(denoised_l, 1.30, blurred, -0.30, 0), 0, 255).astype(np.uint8)

            denoised_a = cv2.bilateralFilter(a_chan, d=5, sigmaColor=20, sigmaSpace=20)
            denoised_b = cv2.bilateralFilter(b_chan, d=5, sigmaColor=20, sigmaSpace=20)
            return cv2.cvtColor(cv2.merge((sharp_l, denoised_a, denoised_b)), cv2.COLOR_LAB2RGB)

    def _tiled_forward_pass(self, model, image_rgb, patch_size=256, stride=192):
        """
        Processes high-resolution images via sliding-window patches with 2D Hann window blending.
        Eliminates boundary seam artifacts and ensures constant memory footprint on CPU/GPU.
        """
        h, w = image_rgb.shape[:2]

        # For small images, run standard padded forward pass
        if h <= patch_size and w <= patch_size:
            pad_h = patch_size - h
            pad_w = patch_size - w
            padded = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            tensor_in = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_t = model(tensor_in)
            out_np = (torch.clamp(out_t[0], 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            return out_np[:h, :w]

        # 2D Hann window for smooth multi-patch blending
        y_win = np.hanning(patch_size)
        x_win = np.hanning(patch_size)
        blend_weight = np.outer(y_win, x_win)[:, :, np.newaxis].astype(np.float32)

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        weight_canvas = np.zeros((h, w, 1), dtype=np.float32) + 1e-6

        y_steps = list(range(0, h - patch_size, stride))
        if not y_steps or y_steps[-1] != h - patch_size:
            y_steps.append(max(0, h - patch_size))

        x_steps = list(range(0, w - patch_size, stride))
        if not x_steps or x_steps[-1] != w - patch_size:
            x_steps.append(max(0, w - patch_size))

        for y in y_steps:
            for x in x_steps:
                patch = image_rgb[y:y+patch_size, x:x+patch_size]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    patch = cv2.resize(patch, (patch_size, patch_size))
                patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    patch_out = model(patch_tensor)
                patch_np = (torch.clamp(patch_out[0], 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.float32)

                canvas[y:y+patch_size, x:x+patch_size] += patch_np * blend_weight
                weight_canvas[y:y+patch_size, x:x+patch_size] += blend_weight

        blended = np.clip(canvas / weight_canvas, 0, 255).astype(np.uint8)
        return blended

    def restore_image(self, image_input, ground_truth_path=None, run_ocr=True, target_lang="en", model_name="best", domain="palm_leaf"):
        """
        End-to-End Heritage Restoration, Metric Benchmarking, Text Extraction, and Translation.
        image_input: filepath (str) or numpy array (RGB/BGR/Grayscale/RGBA)
        """
        t0 = time.time()

        # 1. Normalize input image format
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise FileNotFoundError(f"Cannot read image at: {image_input}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            source_filename = os.path.basename(image_input)
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                image_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                image_rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            else:
                image_rgb = arr
            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb * 255.0 if image_rgb.max() <= 1.0 else image_rgb, 0, 255).astype(np.uint8)
            source_filename = f"upload_{int(time.time())}.png"
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        h_orig, w_orig = image_rgb.shape[:2]

        # Cloud/CPU deployment speed and memory protection:
        # Cap working resolution to 512px on CPU so neural restoration completes in seconds
        max_work_dim = 512 if self.device.type == "cpu" else 1024
        if max(h_orig, w_orig) > max_work_dim:
            scale = float(max_work_dim) / max(h_orig, w_orig)
            working_rgb = cv2.resize(image_rgb, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        else:
            working_rgb = image_rgb.copy()

        h, w = working_rgb.shape[:2]

        # 2. Determine target domain
        if not domain or domain.lower() in ["auto", "detect", "default"]:
            resolved_domain = self._auto_detect_domain(working_rgb)
        else:
            resolved_domain = domain.lower()

        # 3. Retrieve model and execute neural forward pass
        model, actual_model_name = self.get_model(model_name)

        if actual_model_name == "DnCNN":
            # Fully convolutional: process directly at native resolution
            pad_h = (16 - (h % 16)) % 16
            pad_w = (16 - (w % 16)) % 16
            padded = cv2.copyMakeBorder(working_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            norm_tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_tensor = model(norm_tensor)
            out_np = (torch.clamp(out_tensor[0], 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            restored_raw = out_np[:h, :w]
        elif max(h, w) <= 512:
            # Single forward pass for documents up to 512px (instantaneous on CPU)
            pad_h = (16 - (h % 16)) % 16
            pad_w = (16 - (w % 16)) % 16
            padded = cv2.copyMakeBorder(working_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            norm_tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_tensor = model(norm_tensor)
            out_np = (torch.clamp(out_tensor[0], 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            restored_raw = out_np[:h, :w]
        else:
            # Sliding-window Hann tiled inference for large documents
            restored_raw = self._tiled_forward_pass(model, working_rgb, patch_size=256, stride=192)

        # Scale back to original dimensions if downscaled
        if (h, w) != (h_orig, w_orig):
            restored_raw = cv2.resize(restored_raw, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        # 4. Domain-Adaptive Archival Enhancement
        restored_final = self._apply_domain_adaptive_enhancement(image_rgb, restored_raw, domain=resolved_domain)

        # 5. Save Restored Output
        run_id = str(uuid.uuid4())
        restored_filename = f"{run_id}_restored.png"
        restored_disk_path = os.path.join(OUTPUTS_DIR, restored_filename)
        cv2.imwrite(restored_disk_path, cv2.cvtColor(restored_final, cv2.COLOR_RGB2BGR))
        restored_url = f"/static/outputs/{restored_filename}"

        # 6. Honest Metric Evaluation
        metrics = None
        metrics_message = "Reference image unavailable — quantitative metrics cannot be calculated."
        has_ground_truth = False

        if ground_truth_path and os.path.exists(ground_truth_path):
            gt_bgr = cv2.imread(ground_truth_path)
            if gt_bgr is not None:
                gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
                if gt_rgb.shape[:2] != (h_orig, w_orig):
                    gt_rgb = cv2.resize(gt_rgb, (w_orig, h_orig))

                rest_norm = restored_final.astype(np.float32) / 255.0
                gt_norm = gt_rgb.astype(np.float32) / 255.0

                psnr_val = compute_psnr(gt_norm, rest_norm)
                ssim_val = compute_ssim(gt_norm, rest_norm)
                mse_val = float(np.mean((gt_norm - rest_norm) ** 2))
                mae_val = float(np.mean(np.abs(gt_norm - rest_norm)))

                metrics = {
                    "PSNR": float(round(float(psnr_val), 2)),
                    "SSIM": float(round(float(ssim_val), 4)),
                    "MSE": float(round(float(mse_val), 4)),
                    "MAE": float(round(float(mae_val), 4))
                }
                metrics_message = f"PSNR: {metrics['PSNR']} dB | SSIM: {metrics['SSIM']} | MSE: {metrics['MSE']} | MAE: {metrics['MAE']}"
                has_ground_truth = True

        # 7. Multi-pass OCR Text Extraction
        extracted_text = ""
        # Free neural restoration model and intermediate tensors to ensure maximum headroom for OCR
        try:
            del model
            self._model_cache.clear()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if run_ocr:
            try:
                extracted_text = extract_text_from_heritage(restored_final)
            except Exception as e:
                print(f"[!] OCR Notice: {e}")
                extracted_text = ""
            if not extracted_text:
                extracted_text = "No inscribed text detected in this scan region."

        # 8. Translation
        translated_text = ""
        if extracted_text and not extracted_text.startswith("No inscribed"):
            try:
                translated_text = translate_heritage_text(extracted_text, target_lang=target_lang)
            except Exception as e:
                print(f"[!] Translation Notice: {e}")
                translated_text = "N/A"
        else:
            translated_text = "N/A"

        elapsed = round(time.time() - t0, 2)

        return {
            "success": True,
            "run_id": run_id,
            "model_name": actual_model_name,
            "domain": resolved_domain,
            "restored_path": restored_disk_path,
            "restored_url": restored_url,
            "has_ground_truth": has_ground_truth,
            "metrics": metrics,
            "metrics_message": metrics_message,
            "extracted_text": extracted_text,
            "translated_text": translated_text,
            "latency_sec": elapsed,
            "processing_time": elapsed
        }

# Global singleton
_restorer_instance = None

def get_restorer():
    global _restorer_instance
    if _restorer_instance is None:
        _restorer_instance = HeritageRestorer()
    return _restorer_instance

def restore_image(image_input, ground_truth_path=None, run_ocr=True, target_lang="en", model_name="best", domain="palm_leaf"):
    restorer = get_restorer()
    return restorer.restore_image(image_input, ground_truth_path=ground_truth_path,
                                  run_ocr=run_ocr, target_lang=target_lang,
                                  model_name=model_name, domain=domain)

def translate_text(text, target_lang="en"):
    return translate_heritage_text(text, target_lang=target_lang)

if __name__ == "__main__":
    restorer = get_restorer()
    sample = os.path.join(WORKSPACE_DIR, "test_samples", "external_palmleaf_grantha_01.png")
    if not os.path.exists(sample):
        sample = os.path.join(WORKSPACE_DIR, "static", "images", "warm", "hero_manuscript.png")
    if os.path.exists(sample):
        print(f"Testing inference on: {sample}")
        res = restorer.restore_image(sample, ground_truth_path=None, model_name="Restormer", domain="palm_leaf")
        print("Model Used:", res["model_name"])
        print("Restored URL:", res["restored_url"])
        print("Metrics Status:", res["metrics_message"])
        # Safe ASCII representation of extracted text
        safe_repr = res["extracted_text"][:40].encode("ascii", "replace").decode("ascii")
        print("Extracted Text Preview:", safe_repr)
