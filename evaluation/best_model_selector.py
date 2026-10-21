import os
import sys

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

import json
import shutil
import pandas as pd
import numpy as np

OUTPUTS_DIR = os.path.join(WORKSPACE_DIR, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(OUTPUTS_DIR, "best_model")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

def select_and_deploy_best_model():
    print("=" * 75)
    print(" Selecting and Deploying Best Model Based on Real Test Evaluation")
    print("=" * 75)

    comp_json_path = os.path.join(OUTPUTS_DIR, "model_comparison.json")
    if not os.path.exists(comp_json_path):
        raise FileNotFoundError(f"Model comparison results not found at {comp_json_path}. Run evaluation first.")

    with open(comp_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    models = list(results.keys())
    if not models:
        raise ValueError("No models found in comparison results.")

    # Extract raw arrays for normalization
    psnrs = [results[m]["PSNR"] for m in models]
    ssims = [results[m]["SSIM"] for m in models]
    lpipss = [results[m]["LPIPS"] for m in models]
    cers = [results[m]["CER"] for m in models]
    wers = [results[m]["WER"] for m in models]
    times = [results[m]["Inference_Time_ms"] for m in models]

    # Normalizations: (val - min) / (max - min)
    def normalize(vals, higher_is_better=True):
        v_min, v_max = min(vals), max(vals)
        if v_max == v_min:
            return [1.0 for _ in vals]
        if higher_is_better:
            return [(v - v_min) / (v_max - v_min) for v in vals]
        return [(v_max - v) / (v_max - v_min) for v in vals]

    norm_psnr = normalize(psnrs, higher_is_better=True)
    norm_ssim = normalize(ssims, higher_is_better=True)
    norm_lpips = normalize(lpipss, higher_is_better=False)  # lower LPIPS is better
    norm_cer = normalize(cers, higher_is_better=False)      # lower CER is better
    norm_wer = normalize(wers, higher_is_better=False)      # lower WER is better
    norm_speed = normalize(times, higher_is_better=False)   # lower latency is better

    # Composite Score Weights
    # PSNR = 20%, SSIM = 25%, LPIPS = 20%, CER = 20%, WER = 10%, Runtime = 5%
    w_psnr, w_ssim, w_lpips, w_cer, w_wer, w_speed = 0.20, 0.25, 0.20, 0.20, 0.10, 0.05

    scores = {}
    for i, m in enumerate(models):
        score = (
            w_psnr * norm_psnr[i] +
            w_ssim * norm_ssim[i] +
            w_lpips * norm_lpips[i] +
            w_cer * norm_cer[i] +
            w_wer * norm_wer[i] +
            w_speed * norm_speed[i]
        )
        scores[m] = round(float(score) * 100.0, 2)  # 0 - 100 score
        results[m]["Overall_Score"] = scores[m]

    # Update CSV with overall score
    csv_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["Overall_Score"] = [scores.get(m, 0.0) for m in df["Model"]]
        df = df.sort_values(by="Overall_Score", ascending=False)
        df.to_csv(csv_path, index=False)

    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Determine individual category winners
    best_overall = max(scores, key=scores.get)
    best_psnr = models[int(np.argmax(psnrs))]
    best_ssim = models[int(np.argmax(ssims))]
    best_lpips = models[int(np.argmin(lpipss))]
    best_ocr = models[int(np.argmin(cers))]
    fastest = models[int(np.argmin(times))]

    print(f"\nRANKING & BENCHMARK LEADERS:")
    print(f"  [*] BEST OVERALL MODEL: {best_overall} (Composite Score: {scores[best_overall]}/100)")
    print(f"  - Best PSNR Model:    {best_psnr} ({results[best_psnr]['PSNR']} dB)")
    print(f"  - Best SSIM Model:    {best_ssim} ({results[best_ssim]['SSIM']})")
    print(f"  - Best LPIPS Model:   {best_lpips} ({results[best_lpips]['LPIPS']})")
    print(f"  - Best OCR Accuracy:  {best_ocr} (CER: {results[best_ocr]['CER']})")
    print(f"  - Fastest Model:      {fastest} ({results[fastest]['Inference_Time_ms']} ms)")

    # Deploy best model to outputs/best_model/
    clean_best_name = best_overall.lower().replace(" ", "_").replace("-", "")
    src_ckpt = os.path.join(CHECKPOINTS_DIR, clean_best_name, "best_generator.pth" if "pix2pix" in clean_best_name else "best.pth")
    src_cfg = os.path.join(CHECKPOINTS_DIR, clean_best_name, "config.json")

    dst_ckpt = os.path.join(BEST_MODEL_DIR, "model.pth")
    dst_cfg = os.path.join(BEST_MODEL_DIR, "model_config.json")
    dst_metrics = os.path.join(BEST_MODEL_DIR, "metrics.json")
    dst_readme = os.path.join(BEST_MODEL_DIR, "README.txt")

    if os.path.exists(src_ckpt):
        shutil.copy2(src_ckpt, dst_ckpt)
    else:
        print(f"Warning: Checkpoint {src_ckpt} not found on disk.")

    if os.path.exists(src_cfg):
        shutil.copy2(src_cfg, dst_cfg)
    else:
        with open(dst_cfg, "w") as f:
            json.dump({"model_name": best_overall}, f)

    with open(dst_metrics, "w", encoding="utf-8") as f:
        json.dump(results[best_overall], f, indent=4)

    # Write README.txt
    metrics_info = results[best_overall]
    with open(dst_readme, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" BEST DIGITAL HERITAGE RESTORATION MODEL: {best_overall}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model Name:          {best_overall}\n")
        f.write(f"Model Version:       1.0.0 (Trained Checkpoint)\n")
        f.write(f"Dataset Version:     Multi-Domain Heritage Dataset (3,000 samples)\n")
        f.write(f"Deployment Date:     {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("BENCHMARK METRICS (HELD-OUT TEST SET):\n")
        f.write(f"  * PSNR:            {metrics_info['PSNR']} dB\n")
        f.write(f"  * SSIM:            {metrics_info['SSIM']}\n")
        f.write(f"  * LPIPS:           {metrics_info['LPIPS']}\n")
        f.write(f"  * MAE:             {metrics_info['MAE']}\n")
        f.write(f"  * MSE:             {metrics_info['MSE']}\n")
        f.write(f"  * OCR CER:         {metrics_info['CER']}\n")
        f.write(f"  * OCR WER:         {metrics_info['WER']}\n")
        f.write(f"  * Latency:         {metrics_info['Inference_Time_ms']} ms\n")
        f.write(f"  * Overall Score:   {scores[best_overall]} / 100\n\n")
        f.write("SELECTION REASON:\n")
        f.write(f"Selected automatically as the top performer across image fidelity, structural retention,\n")
        f.write(f"character stroke readability (OCR CER/WER), and runtime performance.\n")

    print(f"\nBest model successfully deployed to: {BEST_MODEL_DIR}")
    return best_overall, results

if __name__ == "__main__":
    select_and_deploy_best_model()
