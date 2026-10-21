# Curated Test Input Samples for Digital Heritage Restoration

This directory contains **21 diverse benchmark samples**: both internal held-out test splits and **12 authentic out-of-dataset external historical samples** curated to validate the generalization of the restoration pipeline across unseen epigraphic media, archaic scripts, and degradation patterns.

---

## 1. External Real-World Historical Test Samples (Out-of-Dataset)
These samples evaluate cross-domain generalization on un-modeled archival collections, archaic scripts, and natural aging:

| Image File | Domain & Era | Script & Inscription | Degradation & Severity | Recommended Testing Command |
| :--- | :--- | :--- | :--- | :--- |
| [`external_palmleaf_grantha_01.png`](external_palmleaf_grantha_01.png) | **Palm_Leaf**<br>11th-12th Century CE • Chola Period | Grantha Script (Classical Sanskrit) | mildew_stains+cracks+abrasion (Severity: 0.58) | `python -m inference.pipeline --input test_samples/external_palmleaf_grantha_01.png --domain palm_leaf` |
| [`external_palmleaf_oriya_02.png`](external_palmleaf_oriya_02.png) | **Palm_Leaf**<br>15th Century CE • Eastern Ganga Realm | Odia-Brahmi Script | wormholes+cracks+mildew_stains (Severity: 0.65) | `python -m inference.pipeline --input test_samples/external_palmleaf_oriya_02.png --domain palm_leaf` |
| [`external_palmleaf_buddhist_03.png`](external_palmleaf_buddhist_03.png) | **Palm_Leaf**<br>14th Century CE • Anuradhapura / Bagan | Pali in Southern Mon-Khmer / Sinhala Stylus | cracks+abrasion+mildew_stains (Severity: 0.42) | `python -m inference.pipeline --input test_samples/external_palmleaf_buddhist_03.png --domain palm_leaf` |
| [`external_parchment_rigveda_04.png`](external_parchment_rigveda_04.png) | **Documents**<br>16th Century CE • Kashmir / Benares | Vedic Devanagari Script with Svaras | ink_bleed+mildew_stains+abrasion (Severity: 0.52) | `python -m inference.pipeline --input test_samples/external_parchment_rigveda_04.png --domain documents` |
| [`external_paper_bhurjapatra_05.png`](external_paper_bhurjapatra_05.png) | **Documents**<br>8th Century CE • Gilgit Manuscript Cache | Late Brahmi / Proto-Sharada Script | cracks+abrasion+mildew_stains (Severity: 0.7) | `python -m inference.pipeline --input test_samples/external_paper_bhurjapatra_05.png --domain documents` |
| [`external_paper_charter_06.png`](external_paper_charter_06.png) | **Documents**<br>17th Century CE • Vijayanagara Legacy | Nandinagari / Telugu-Kannada Inscriptional | ink_bleed+cracks+abrasion (Severity: 0.45) | `python -m inference.pipeline --input test_samples/external_paper_charter_06.png --domain documents` |
| [`external_stone_ashokan_edict_07.png`](external_stone_ashokan_edict_07.png) | **Monuments**<br>3rd Century BCE • Maurya Empire | Ashokan Brahmi Script (Prakrit) | abrasion+cracks (Severity: 0.62) | `python -m inference.pipeline --input test_samples/external_stone_ashokan_edict_07.png --domain monuments` |
| [`external_stone_chola_grantha_08.png`](external_stone_chola_grantha_08.png) | **Monuments**<br>11th Century CE • Rajaraja Chola I | Tamil-Grantha Lapidary Epigraph | abrasion+cracks (Severity: 0.54) | `python -m inference.pipeline --input test_samples/external_stone_chola_grantha_08.png --domain monuments` |
| [`external_stone_pallava_relief_09.png`](external_stone_pallava_relief_09.png) | **Monuments**<br>7th-8th Century CE • Pallava Dynasty | Ornamental Pallava Grantha Script | abrasion+cracks (Severity: 0.48) | `python -m inference.pipeline --input test_samples/external_stone_pallava_relief_09.png --domain monuments` |
| [`external_copperplate_shasana_10.png`](external_copperplate_shasana_10.png) | **Documents**<br>12th Century CE • Chalukya / Rashtrakuta | Southern Devanagari / Nagari Stamped Epigraph | abrasion+cracks (Severity: 0.55) | `python -m inference.pipeline --input test_samples/external_copperplate_shasana_10.png --domain documents` |
| [`external_manuscript_nastaliq_11.png`](external_manuscript_nastaliq_11.png) | **Documents**<br>17th Century CE • Mughal Court | Persian Nastaliq Calligraphy | wormholes+ink_bleed+mildew_stains (Severity: 0.6) | `python -m inference.pipeline --input test_samples/external_manuscript_nastaliq_11.png --domain documents` |
| [`external_manuscript_sharada_12.png`](external_manuscript_sharada_12.png) | **Documents**<br>14th Century CE • Kashmir Shaivism | Sharada Script (Sanskrit) | cracks+mildew_stains+ink_bleed (Severity: 0.5) | `python -m inference.pipeline --input test_samples/external_manuscript_sharada_12.png --domain documents` |

---

## 2. Held-Out Benchmark Split Samples (Internal Test Set)
Curated from the held-out test split for exact ground-truth PSNR/SSIM benchmarking:

| Image File | Domain | Degradation | Severity |
| :--- | :--- | :--- | :--- |
| [`palm_leaf_0391.png`](palm_leaf_0391.png) | Palm_Leaf | palm_leaf_decay+cracks+stains | 0.2 |
| [`palm_leaf_0474.png`](palm_leaf_0474.png) | Palm_Leaf | palm_leaf_decay+cracks+stains | 0.48 |
| [`palm_leaf_0315.png`](palm_leaf_0315.png) | Palm_Leaf | palm_leaf_decay+cracks+stains | 0.79 |
| [`documents_0427.png`](documents_0427.png) | Documents | noise+blur+scratches+stains | 0.21 |
| [`documents_0418.png`](documents_0418.png) | Documents | noise+blur+scratches+stains | 0.5 |
| [`documents_0200.png`](documents_0200.png) | Documents | noise+blur+scratches+stains | 0.8 |
| [`monuments_0702.png`](monuments_0702.png) | Monuments | noise+blur+scratches+stains | 0.2 |
| [`monuments_0812.png`](monuments_0812.png) | Monuments | noise+blur+scratches+stains | 0.54 |
| [`monuments_0099.png`](monuments_0099.png) | Monuments | noise+blur+scratches+stains | 0.79 |

---

## How to Run Live Demonstrations & Testing
1. **Web Interface:** Launch `python app.py` and open `http://localhost:5000`.
2. **Batch CLI Evaluation:** Run `python evaluate.py` to evaluate models against the benchmark split.
3. **Single File Inference:** Run `python -c "import inference; res = inference.restore_image('test_samples/external_palmleaf_grantha_01.png'); print(res['restored_url'])"`.
