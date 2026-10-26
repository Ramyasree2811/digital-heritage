# 📜 Digital Heritage Restoration Platform
### *Deep Learning-Powered Restoration, Epigraphic OCR Extraction & Multilingual Translation for Ancient Manuscripts and Inscriptions*

---

## 🏛️ Project Overview: What We Built

Ancient cultural heritage assets—such as palm-leaf manuscripts, historical charters, and stone inscriptions—suffer from severe physical degradation over centuries. Common defects include:
* **Fungal mildew and water stains** that obscure written text.
* **Wormholes, cracks, and physical tears** causing complete loss of structural material.
* **Ink bleed-through and faded calligraphy** reducing stroke readability.
* **Weathering and surface abrasion** eroding carved stone lapidary epigraphs.

To solve this challenge, we built an **end-to-end Digital Heritage Restoration Platform**. The platform takes degraded historical scans and executes an integrated 5-stage preservation pipeline:

```
[Degraded Scan] ➔ [Damage Mask Detection] ➔ [Deep Learning Restoration] ➔ [2x Super-Resolution] ➔ [Multi-Pass OCR] ➔ [Multilingual Translation]
```

The system preserves character stroke integrity, enhances readability, transcribes archaic and classical scripts (Sanskrit, Devanagari, Grantha, Hindi, English), and translates the inscribed wisdom into modern languages in real time.

---

## ⚙️ Core Pipeline Components

### 1. Text-Preserving Damage Mask Detection
* Uses multi-scale morphological background analysis and adaptive thresholding to identify genuine structural defects (wormholes, tears, paper cracks, damp stains).
* **Text Stroke Protection:** Automatically isolates inscribed text characters from the defect mask so inpainting algorithms restore background parchment **without erasing or blurring ancient writing**.

### 2. Deep Learning Restoration Engine
* Houses **5 distinct neural network architectures** trained over **100 epochs** on heritage datasets:
  1. **Restormer** *(Selected Best Model)*: Multi-Dconv head transposed attention architecture designed for high-resolution image restoration without boundary seam artifacts.
  2. **SwinIR**: Image restoration utilizing shifted-window Swin Transformer backbones.
  3. **U-Net**: Multi-scale encoder-decoder architecture with dense skip connections to preserve localized edge details.
  4. **Pix2Pix (cGAN)**: Conditional Generative Adversarial Network for paired image-to-image translation.
  5. **DnCNN**: Deep feed-forward convolutional neural network for residual noise and background stain reduction.

### 3. 2× Super-Resolution Enhancement
* Upscales the restored document to double its spatial resolution, sharpening character edges and enhancing fine ink strokes for archival viewing and scholarly transcription.

### 4. High-Accuracy Epigraphic OCR Engine
* Multi-pass character recognition tailored for weathered epigraphs using EasyOCR with:
  * Morphological background division normalization: $\frac{I}{I_{\text{bg}}} \times 240$ to eliminate palm-leaf wood grain and yellowed parchment.
  * Contrast Limited Adaptive Histogram Equalization (CLAHE) for ink edge sharpening.
  * Multi-core parallel execution delivering transcription in **under 4 seconds**.

### 5. Fast Multilingual Translation Cascade
* Translates transcribed Sanskrit, Hindi, and regional epigraphs into **English, Hindi, Tamil, Telugu, Sanskrit, French, German, and Spanish**.
* **Indic Continuous Script Boundary Segmentation:** Inserts natural word boundaries into unspaced Sanskrit/Devanagari text while preserving combining vowel signs (*matras* and *viramas*).
* **Morphological Root Synthesis:** Decomposes inflected classical Sanskrit words into grammatical roots and suffixes to decipher weathered or previously uncataloged inscriptions.
* Returns translations in **0.15s – 0.5s**.

---

## 📊 Model Evaluation & 100-Epoch Benchmark Results

All five architectures were systematically benchmarked on the held-out test split across 100 epochs using standard image reconstruction metrics:
* **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction fidelity (higher is better).
* **SSIM (Structural Similarity Index)**: Measures structural and perceptual preservation (closer to 1.0 is better).
* **MSE (Mean Squared Error)** & **MAE (Mean Absolute Error)**: Quantifies pixel-level deviation.

| Model Architecture | PSNR (dB) ↑ | SSIM ↑ | MSE ↓ | MAE ↓ | Inference Latency | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Restormer** | **19.16 dB** | **0.6419** | **0.0156** | **0.0651** | 610 ms | 🏆 **Active Production Model** |
| **SwinIR** | 18.84 dB | 0.6280 | 0.0171 | 0.0694 | 740 ms | Evaluated Checkpoint |
| **U-Net** | 18.21 dB | 0.6014 | 0.0195 | 0.0742 | 320 ms | Evaluated Checkpoint |
| **Pix2Pix (cGAN)** | 17.65 dB | 0.5892 | 0.0210 | 0.0789 | 450 ms | Evaluated Checkpoint |
| **DnCNN** | 16.92 dB | 0.5510 | 0.0245 | 0.0863 | **180 ms** | Evaluated Checkpoint |

*All training curves, per-epoch logs (1–100), and metric summaries are preserved in `outputs/checkpoints/` and `outputs/model_comparison.csv`.*

---

## 🖼️ Supported Heritage Domains & Curated Datasets

The platform is designed to handle diverse historical media across three primary domains:

| Domain | Historical Media Types | Typical Degradations Addressed |
| :--- | :--- | :--- |
| **🌴 Palm Leaf** | Chola Grantha folios, Odia palm leaves, Southern Buddhist Pali texts | Wood-grain cracks, edge fraying, fungal blotches, faded stylus grooves |
| **📜 Archival Documents** | Vedic Sanskrit parchments, Gilgit birch bark (Bhurjapatra), Vijayanagara charters | Ink bleed-through, paper foxing, water stains, mold spots, brittle tears |
| **🏛️ Monuments** | Ashokan rock edicts, Pallava reliefs, Chola temple wall inscriptions | Stone surface abrasion, erosion, micro-fractures, chisel weathering |

*The repository includes **21 curated benchmark samples** in `test_samples/` (including 12 authentic real-world external historical folios) to validate zero-shot generalization on un-modeled archival collections.*

---

## 💻 Web Application Features

The interactive web application provides a complete digital laboratory for archivists, epigraphists, and researchers:

1. **Before/After Split-View Comparison:**
   * Interactive drag-and-drop image upload with real-time horizontal comparison slider to visually inspect defect removal.
2. **Live Execution Tracker:**
   * Step-by-step progress indicator tracking Damage Masking $\rightarrow$ Neural Inpainting $\rightarrow$ OCR Transcription $\rightarrow$ Translation.
3. **Editable Transcription & Live Translation:**
   * Transcribed OCR text appears in an interactive workspace where users can edit or enter custom text and trigger instant re-translation across 8 target languages.
4. **Audit History & Search:**
   * Persistent database of all processed scans with timestamped records, thumbnail previews, model used, and search filtering.
5. **Academic Report Generation (`.docx`):**
   * One-click download of a structured research report containing image comparisons, defect statistics, OCR transcripts, translations, and quantitative quality metrics.

---

## 📁 Project Structure

```
digital-heritage/
├── app.py                         # Flask web application & REST API routes
├── requirements.txt               # Project library dependencies
├── PROJECT_FILE_OUTPUT_MAPPING.md # Evaluation and viva reference mapping
│
├── inference/                     # Unified Inference & Processing Subsystem
│   ├── ocr_engine.py              # Multi-pass EasyOCR character transcription
│   ├── translation.py            # Neural translation cascade & Sanskrit root lexicon
│   ├── pipeline.py                # End-to-end pipeline coordinator
│   ├── restore.py                 # Tile-blended neural restoration
│   └── super_resolution.py       # 2x image upscaling module
│
├── models/                        # Serialized PyTorch Weights
│   ├── best_model/                # Active production model (Restormer)
│   ├── restormer/                 # Restormer weights & training logs
│   ├── swinir/                    # SwinIR weights & training logs
│   ├── unet/                      # U-Net weights & training logs
│   ├── pix2pix/                   # Pix2Pix weights & training logs
│   └── dncnn/                     # DnCNN weights & training logs
│
├── test_samples/                  # Curated Historical Benchmark Test Samples
│   ├── index.json                 # Metadata mapping domain, era, and severity
│   ├── external_palmleaf_*.png    # Chola & Ganga dynasty Grantha/Odia palm leaves
│   ├── external_parchment_*.png   # 16th Century Rigvedic Sanskrit manuscripts
│   ├── external_stone_*.png       # Ashokan Brahmi & Pallava relief lapidary epigraphs
│   └── README.md                  # Sample evaluation documentation
│
├── static/                        # Frontend UI Assets
│   ├── css/                       # Stylesheets & theme styling
│   ├── js/                        # Split-slider, AJAX controllers
│   ├── uploads/                   # Staged input images
│   └── outputs/                   # Restored archival outputs & damage masks
│
├── templates/                     # Application HTML Pages
│   ├── index.html                 # Main dashboard with split-slider viewer
│   ├── process.html               # Live execution progress bar
│   ├── result.html                # Inspection & translation workspace
│   └── comparison.html            # Multi-model side-by-side benchmark viewer
│
├── training/                      # Training scripts & loss functions (Epochs 1-100)
└── evaluation/                    # Quantitative evaluation & comparison scripts
```

---

## 🛠️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Ramyasree2811/digital-heritage.git
cd digital-heritage
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python app.py
```
Open your web browser and go to:
```
http://127.0.0.1:5000
```

---

## 🎯 Verification & Testing Commands

To test the system from the terminal without the web UI:

* **Run End-to-End Restoration on a Palm-Leaf Sample:**
  ```bash
  python -m inference.pipeline --input test_samples/palm_leaf_0315.png --domain palm_leaf --target_lang en
  ```

* **Test Multi-Model Evaluation Script:**
  ```bash
  python evaluate.py
  ```

---

## 📜 Citation & Academic Use

If you use this system or dataset in your academic research or project presentation:

```bibtex
@misc{digital_heritage_restoration_2026,
  title={Digital Heritage Restoration: Deep Learning Platform for Ancient Manuscript Enhancement, Epigraphic OCR, and Multilingual Translation},
  author={Heritage Restoration Engineering Team},
  year={2026},
  howpublished={\url{https://github.com/Ramyasree2811/digital-heritage}}
}
```
