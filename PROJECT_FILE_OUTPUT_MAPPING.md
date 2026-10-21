# Digital Heritage Restoration — File and Output Mapping Guide

**Project Title:** Digital Heritage Restoration using Deep Learning  
**Domain:** Computer Vision, Deep Learning, Historical Epigraphy & Document Restoration  

---

## 1. Executive Summary

This document outlines the complete architectural mapping between the source codebase scripts, the models trained, the quantitative metric files generated (Epochs 1–100), and the visual outputs produced by the web platform. Use this document as a quick reference during project evaluations, presentations, and viva examinations.

---

## 2. Training & Epoch Progress Mapping

| Source Script / Runner | What It Generates | Saved Destination Path | Purpose / Description |
| :--- | :--- | :--- | :--- |
| `training/train_all.py` / `training/train_*.py` | Model Weights & Checkpoints | `outputs/checkpoints/<model_name>/best_model.pth`<br>`outputs/checkpoints/<model_name>/model.pth` | Contains the serialized PyTorch neural network weights (`state_dict`) for each of the 5 architectures (DnCNN, Restormer, SwinIR, U-Net, Pix2Pix). |
| `training/train_*.py` | Individual Model Epoch Logs (1–100) | `outputs/checkpoints/<model_name>/training_history.csv` | Tracks per-epoch loss and quality metrics for that single model: `train_loss`, `val_loss`, `PSNR`, `SSIM`, `LPIPS`, and `learning_rate` across all 100 epochs. |
| `evaluation/compare_models.py` | Consolidated Multi-Model 1–100 Epochs Comparison | `outputs/model_comparison.csv`<br>`results/model_comparison.csv` | Master comparative dataset containing **all 5 models across all 100 epochs (500 evaluation records)** with PSNR (dB), SSIM, LPIPS, MAE, MSE, latency, and overall scores. |

---

## 3. Evaluation & Qualitative Comparison Mapping

| Source Script / Runner | What It Generates | Saved Destination Path | Purpose / Description |
| :--- | :--- | :--- | :--- |
| `evaluation/compare_models.py` | Benchmark Metrics Summary | `outputs/model_comparison.json` | JSON format key-value evaluation metrics (PSNR, SSIM, LPIPS, CER, WER, Memory, Inference Time) calculated over the held-out test split. |
| `evaluation/compare_models.py` | Multi-Model Qualitative Image Strips | `outputs/qualitative_comparison/compare_documents_*.png` | Visual comparison grid displaying side-by-side patches: **[Degraded Input → Ground Truth → DnCNN → U-Net → Pix2Pix → Restormer → SwinIR]**. |
| `evaluation/select_best_model.py` | Production Deployment Checkpoint | `models/best_model/model.pth`<br>`models/best_model/model_config.json` | Automatically selects and packages the highest-scoring model (Restormer) along with its metadata configuration for the real-time inference web application. |

---

## 4. Live Web Application & Inference Pipeline Mapping

| Component / Endpoint | Generated Output | Output File Path / UI Destination | Purpose / Description |
| :--- | :--- | :--- | :--- |
| `app.py` (`POST /api/restore`) | Uploaded User Document | `static/uploads/<uuid>.<ext>` | Raw historical scans (palm leaf, paper manuscript, or stone inscription) uploaded by users from the browser. |
| `inference.py` (`HeritageRestorer`) | Neural Restored Image | `static/outputs/<uuid>_restored.png` | Archival restored image produced by the active neural network model combined with adaptive bilateral background normalization. |
| `inference/ocr_engine.py` | Extracted Inscribed Text | JSON API field `extracted_text`<br>UI textarea: `#ocr-text-content` | Multi-pass OCR transcription (Devanagari, Sanskrit, Latin, or Chinese) with CLAHE and Otsu contrast enhancement. |
| `inference/translation.py` | Multilingual Translation | JSON API field `translated_text`<br>UI box: `#translated-output-box` | Translation output into English, Hindi, Tamil, Telugu, etc., powered by Google GTX unthrottled endpoint and scholarly scripture databases. |
| `app.py` | Session History & Audit Trail | `outputs/history.json` | JSON database storing every processed document, timestamp, input/output image URLs, model used, quality metrics, and transcriptions. |

---

## 5. Guide Q&A Reference Sheet

When explaining the system to your guide or external examiner, refer to the following questions and file paths:

### Q1: "Where are the metrics proving that your models trained over 100 epochs?"
> **Answer:** "All 100 epoch evaluations for all 5 models are logged in `outputs/model_comparison.csv`. In addition, per-model training dynamics (train loss, validation loss, PSNR, SSIM, and learning rate schedules) are stored in `outputs/checkpoints/<model_name>/training_history.csv`."

### Q2: "Which model performed the best and how was it selected?"
> **Answer:** "Restormer achieved the highest performance on our test set with a PSNR of 20.13 dB and SSIM of 0.6909. It was automatically selected and deployed via `evaluation/select_best_model.py`, saving the active weights to `models/best_model/model.pth` and configuration to `models/best_model/model_config.json`."

### Q3: "Where can I visually compare the outputs of all 5 models on the same document?"
> **Answer:** "The side-by-side visual strip comparisons are saved in `outputs/qualitative_comparison/`. Each file demonstrates the degraded image, the ground truth, and the reconstructions of all five models."

### Q4: "How does the system handle real user uploads outside the dataset?"
> **Answer:** "The user upload is saved to `static/uploads/`, processed through `inference.py` using our best model, rendered side-by-side with split-slider controls on the web UI, and transcribed using our multi-pass EasyOCR pipeline in `inference/ocr_engine.py`. In addition, 12 authentic out-of-dataset historical test samples (Tanjore Grantha palm leaf, Rigvedic paper folio, Ashokan rock edicts, and copper plate grants) are cataloged in `test_samples/` to demonstrate zero-shot generalization."

### Q5: "How is the project deployed to the cloud?"
> **Answer:** "The system is containerized via `Dockerfile` and configured for deployment on Render, Google Cloud Run, Hugging Face Spaces, and AWS. Full instructions are available in `CLOUD_DEPLOYMENT_GUIDE.md`."
