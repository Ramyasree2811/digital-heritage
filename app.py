import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import uuid
import time
import shutil
import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, send_from_directory

# Import the unified restoration engine
from inference import HeritageRestorer, translate_heritage_text
from inference.pipeline import run_heritage_pipeline
from inference.restore import restore_heritage_image

restorer = HeritageRestorer()

# Pre-warm OCR engine in a background daemon thread to eliminate cold-start lag
import threading
def _prewarm_ocr():
    try:
        from inference.ocr_engine import init_ocr
        init_ocr()
    except Exception as e:
        print(f'Notice during background OCR pre-warm: {e}')

threading.Thread(target=_prewarm_ocr, daemon=True).start()

app = Flask(__name__)

# Paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(WORKSPACE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(WORKSPACE_DIR, "static", "outputs")
HISTORY_FILE = os.path.join(WORKSPACE_DIR, "outputs", "history.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M'):
    import datetime
    try:
        return datetime.datetime.fromtimestamp(value).strftime(format)
    except Exception:
        return "Recent"


# Keep track of active pipeline progress in memory (keyed by run_id)
# Maps run_id -> {"step": str, "progress": int, "completed": bool}
progress_tracker = {}

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "digital-heritage-restoration"}), 200

@app.route("/")
def index():
    history = load_history()
    sorted_history = sorted(history.items(), key=lambda x: x[1].get("timestamp", 0), reverse=True)
    return render_template("index.html", history=sorted_history)

@app.route("/api/restore", methods=["POST"])
def api_restore():
    if 'image' not in request.files and not request.form.get("sample_image"):
        return jsonify({"success": False, "error": "No image provided"}), 400
        
    model_name = request.form.get("model", "best")
    domain = request.form.get("domain", "palm_leaf")
    target_lang = request.form.get("target_lang", "en")
    run_ocr = request.form.get("run_ocr", "true").lower() in ["true", "on", "1"]

    run_id = str(uuid.uuid4())
    is_user_upload = False
    gt_path = None
    
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        ext = os.path.splitext(file.filename)[1] or ".png"
        filename = f"{run_id}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        is_user_upload = True
    else:
        sample_rel = request.form.get("sample_image").lstrip("/")
        filepath = os.path.join(WORKSPACE_DIR, sample_rel)
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": f"Sample image {sample_rel} not found"}), 404
        ext = os.path.splitext(filepath)[1] or ".png"
        filename = f"{run_id}{ext}"
        new_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        shutil.copy2(filepath, new_path)
        filepath = new_path
        
        # Check if sample image has a corresponding ground truth clean image
        clean_cand = sample_rel.replace("degraded", "clean")
        clean_full = os.path.join(WORKSPACE_DIR, clean_cand)
        if os.path.exists(clean_full) and "clean" in clean_cand:
            gt_path = clean_full

    try:
        res = restorer.restore_image(
            image_input=filepath,
            ground_truth_path=gt_path,
            run_ocr=run_ocr,
            target_lang=target_lang,
            model_name=model_name,
            domain=domain
        )

        extracted_text = res["extracted_text"]
        translation = res["translated_text"]

        # Update history
        history = load_history()
        history[run_id] = {
            "id": run_id,
            "timestamp": time.time(),
            "original": f"/static/uploads/{filename}",
            "restored": res["restored_url"],
            "model": res["model_name"],
            "domain": domain,
            "has_ground_truth": res["has_ground_truth"],
            "metrics": res.get("metrics"),
            "metrics_message": res["metrics_message"],
            "text": extracted_text,
            "translation": translation
        }
        save_history(history)

        return jsonify({
            "success": True,
            "run_id": run_id,
            "original_url": f"/static/uploads/{filename}",
            "restored_url": res["restored_url"],
            "has_ground_truth": res["has_ground_truth"],
            "metrics": res.get("metrics"),
            "metrics_message": res["metrics_message"],
            "extracted_text": extracted_text,
            "translated_text": translation,
            "model_used": res["model_name"],
            "domain": domain,
            "processing_time": res.get("processing_time") or res.get("latency_sec") or 0.0
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json() or {}
    text = data.get("text", "")
    target_lang = data.get("target_lang", "en")
    translation = translate_heritage_text(text, target_lang=target_lang)
    return jsonify({"success": True, "translation": translation, "target_lang": target_lang})

@app.route("/api/history", methods=["GET"])
def api_history():
    query = request.args.get("q", "").strip().lower()
    filter_type = request.args.get("type", "all").strip().lower()
    
    history = load_history()
    items = list(history.values())
    
    # Sort descending by timestamp
    items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    
    # Filter by type
    filtered = []
    for item in items:
        has_ocr = bool(item.get("text") and len(item.get("text", "").strip()) > 5 and not item.get("text", "").startswith("No inscribed"))
        has_trans = bool(item.get("translation") and len(item.get("translation", "").strip()) > 5 and not item.get("translation", "").startswith("N/A"))
        
        if filter_type == "restored":
            pass # all items in history are restored
        elif filter_type == "ocr":
            if not has_ocr:
                continue
        elif filter_type == "translated":
            if not has_trans:
                continue
                
        # Search query
        if query:
            match_name = query in item.get("id", "").lower() or query in item.get("original", "").lower()
            match_text = query in item.get("text", "").lower()
            match_trans = query in item.get("translation", "").lower()
            match_model = query in item.get("model", "").lower()
            match_domain = query in item.get("domain", "").lower()
            if not (match_name or match_text or match_trans or match_model or match_domain):
                continue
                
        # Add badge info
        badges = ["Restored"]
        if has_ocr:
            badges.append("OCR")
        if has_trans:
            badges.append("Translated")
        item_copy = dict(item)
        item_copy["badges"] = badges
        filtered.append(item_copy)
        
    return jsonify({"success": True, "history": filtered})

@app.route("/api/history/delete/<run_id>", methods=["POST", "DELETE"])
def api_delete_history(run_id):
    history = load_history()
    if run_id in history:
        del history[run_id]
        save_history(history)
        return jsonify({"success": True, "message": "Record deleted"})
    return jsonify({"success": False, "error": "Record not found"}), 404

@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    save_history({})
    return jsonify({"success": True, "message": "History cleared"})

@app.route("/test_samples/<path:filename>")
def serve_test_samples(filename):
    return send_from_directory(os.path.join(WORKSPACE_DIR, "test_samples"), filename)

@app.route("/api/sample_documents", methods=["GET"])
def api_sample_documents():
    samples = [
        {
            "id": "external_grantha_01",
            "title": "Tanjore Grantha Palm-Leaf Folio (External)",
            "period": "11th Century CE • Chola Epigraphy",
            "thumbnail": "/test_samples/external_palmleaf_grantha_01.png",
            "path": "/test_samples/external_palmleaf_grantha_01.png",
            "domain": "palm_leaf",
            "sample_ocr": "ॐ नमो भगवते वासुदेवाय । धर्मो रक्षति रक्षितः सत्यं वद ।",
            "sample_trans": "Salutations to Lord Vasudeva. Dharma protects those who protect it. Speak the truth."
        },
        {
            "id": "external_rigveda_04",
            "title": "Rigvedic Samhita Rag-Paper Folio (External)",
            "period": "16th Century CE • Kashmir / Benares",
            "thumbnail": "/test_samples/external_parchment_rigveda_04.png",
            "path": "/test_samples/external_parchment_rigveda_04.png",
            "domain": "documents",
            "sample_ocr": "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम् । सत्यमेव जयते नानृतम् ॥",
            "sample_trans": "I praise Agni the household priest, the divine minister of the sacrifice. Truth alone triumphs, not untruth."
        },
        {
            "id": "external_ashokan_07",
            "title": "Ashokan Major Rock Edict Inscription (External)",
            "period": "3rd Century BCE • Maurya Basalt Rock",
            "thumbnail": "/test_samples/external_stone_ashokan_edict_07.png",
            "path": "/test_samples/external_stone_ashokan_edict_07.png",
            "domain": "monuments",
            "sample_ocr": "देवानांपिय पियदसि राजा हेवं आहा । धम्मो साधु कियं चु धम्मे ति ॥",
            "sample_trans": "King Priyadarsi, Beloved of the Gods, speaks thus: Dharma is virtuous. What is Dharma? Minimum of sins, many virtues, compassion, and truth."
        },
        {
            "id": "external_copperplate_10",
            "title": "Royal Copper Plate Grant (Tamra-Shasana) (External)",
            "period": "12th Century CE • Chalukya Dynasty",
            "thumbnail": "/test_samples/external_copperplate_shasana_10.png",
            "path": "/test_samples/external_copperplate_shasana_10.png",
            "domain": "documents",
            "sample_ocr": "श्री जयाभ्युदय नृप विक्रमादित्य शके । शासन पत्रं समर्पितम् ॥",
            "sample_trans": "In the auspicious prosperous era of Emperor Vikramaditya, this copper-plate grant is dedicated."
        },
        {
            "id": "sanskrit_manuscript",
            "title": "Ancient Sanskrit Shloka Manuscript",
            "period": "12th Century AD • Rigvedic Palm Leaf",
            "thumbnail": "/static/images/warm/hero_manuscript.png",
            "path": "/static/images/warm/hero_manuscript.png",
            "domain": "documents",
            "sample_ocr": "श्रीगणेशाय नमः । विद्या ददाति विनयं विनयाद् याति पात्रताम् । पात्रत्वात् धनमाप्नोति धनात् धर्मं ततः सुखम् ॥",
            "sample_trans": "Salutations to Lord Ganesha. Knowledge bestows humility; from humility comes worthiness; from worthiness comes wealth; from wealth comes righteousness, and from righteousness comes true happiness."
        },
        {
            "id": "palm_leaf_doc",
            "title": "Palm Leaf Scriptural Folio",
            "period": "14th Century CE • Southern Indic Script",
            "thumbnail": "/static/images/warm/thumb_1.png",
            "path": "/static/images/warm/thumb_1.png",
            "domain": "palm_leaf",
            "sample_ocr": "धर्मो रक्षति रक्षितः सत्यं वद । मातृदेवो भव पितृदेवो भव ।",
            "sample_trans": "Dharma protects those who protect it. Speak the truth. Mother is divine. Father is divine."
        },
        {
            "id": "temple_inscription",
            "title": "Grantha Stone Inscription",
            "period": "Chola Dynasty • 11th Century",
            "thumbnail": "/static/images/warm/thumb_2.png",
            "path": "/static/images/warm/thumb_2.png",
            "domain": "monuments",
            "sample_ocr": "सत्यमेव जयते नानृतम्",
            "sample_trans": "Truth alone triumphs, not untruth."
        }
    ]
    return jsonify({"success": True, "samples": samples})


@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
        
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
        
    model_name = request.form.get("model", "best")
    domain = request.form.get("domain", "palm_leaf")
    run_ocr = request.form.get("run_ocr") == "on"
    run_sr = request.form.get("run_sr") == "on"
    
    if file:
        # Create unique run ID
        run_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        filename = f"{run_id}{ext}"
        
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize progress tracking
        progress_tracker[run_id] = {
            "step": "Initializing Pipeline",
            "progress": 0,
            "completed": False,
            "filepath": filepath,
            "model_name": model_name,
            "domain": domain,
            "run_ocr": run_ocr,
            "run_sr": run_sr
        }
        
        return redirect(url_for("process_screen", run_id=run_id))
        
    return redirect(url_for("index"))

@app.route("/process/<run_id>")
def process_screen(run_id):
    if run_id not in progress_tracker:
        return redirect(url_for("index"))
    return render_template("process.html", run_id=run_id)

@app.route("/process_status/<run_id>")
def process_status(run_id):
    """
    Server-Sent Events (SSE) endpoint to stream pipeline progress updates.
    """
    def event_stream():
        with app.app_context():
            if run_id not in progress_tracker:
                yield f"data: {json.dumps({'error': 'Invalid Run ID'})}\n\n"
                return
                
            tracker = progress_tracker[run_id]
            
            # Callback to update the in-memory tracker during pipeline run
            def status_callback(step, progress):
                tracker["step"] = step
                tracker["progress"] = progress
                
            # We launch the pipeline execution
            try:
                # Setup folders in static for web serving
                web_mask_name = f"{run_id}_damage_mask.png"
                web_restored_name = f"{run_id}_restored.png"
                web_sr_name = f"{run_id}_high_res.png"
                
                # Run pipeline
                pipeline_results = run_heritage_pipeline(
                    image_path=tracker["filepath"],
                    model_name=tracker["model_name"],
                    domain=tracker["domain"],
                    run_ocr=tracker["run_ocr"],
                    run_sr=tracker["run_sr"],
                    status_callback=status_callback
                )
                
                # Copy output files from outputs/pipeline to static/outputs for browser rendering
                shutil.copy2(pipeline_results["mask_path"], os.path.join(OUTPUT_FOLDER, web_mask_name))
                shutil.copy2(pipeline_results["restored_path"], os.path.join(OUTPUT_FOLDER, web_restored_name))
                shutil.copy2(pipeline_results["high_res_path"], os.path.join(OUTPUT_FOLDER, web_sr_name))
                
                # Save to history database
                history = load_history()
                history[run_id] = {
                    "id": run_id,
                    "timestamp": time.time(),
                    "original": f"/static/uploads/{os.path.basename(tracker['filepath'])}",
                    "mask": f"/static/outputs/{web_mask_name}",
                    "restored": f"/static/outputs/{web_restored_name}",
                    "high_res": f"/static/outputs/{web_sr_name}",
                    "model": tracker["model_name"],
                    "domain": tracker["domain"],
                    "text": pipeline_results["extracted_text"],
                    "translation": pipeline_results["translation"]
                }
                save_history(history)
                
                # Clean up progress tracker
                tracker["completed"] = True
                
                # Yield final completion step
                redirect_url = f"/result/{run_id}"
                yield f"data: {json.dumps({'step': 'Pipeline Completed!', 'progress': 100, 'redirect': redirect_url})}\n\n"
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in pipeline execution for run {run_id}: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/result/<run_id>")
def result(run_id):
    history = load_history()
    record = history.get(run_id)
    if not record:
        return redirect(url_for("index"))
    return render_template("result.html", record=record)

@app.route("/compare/<run_id>")
def compare(run_id):
    """
    Runs all 5 models on the uploaded image and displays a side-by-side comparison screen.
    """
    history = load_history()
    record = history.get(run_id)
    if not record:
        return redirect(url_for("index"))
        
    orig_path = os.path.join(WORKSPACE_DIR, record["original"].lstrip("/"))
    mask_path = os.path.join(WORKSPACE_DIR, record["mask"].lstrip("/"))
    
    # Models to compare
    models = ["U-Net", "DnCNN", "Pix2Pix GAN", "Restormer", "SwinIR"]
    comparisons = {}
    
    # Read original image
    image = cv2.imread(orig_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    damage_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    is_doc = (record["domain"] != "monuments")
    
    for m_name in models:
        # Run restore for each model
        restored = restore_heritage_image(image_rgb, model_name=m_name, damage_mask=damage_mask, is_document=is_doc)
        
        # Save to static output folder
        m_filename = f"{run_id}_compare_{m_name.lower().replace(' ', '_')}.png"
        m_path = os.path.join(OUTPUT_FOLDER, m_filename)
        cv2.imwrite(m_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
        
        comparisons[m_name] = f"/static/outputs/{m_filename}"
        
    # Check if metrics json exists
    metrics_json_path = os.path.join(WORKSPACE_DIR, "outputs", "model_comparison.json")
    metrics_data = {}
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            
    return render_template(
        "comparison.html", 
        record=record, 
        comparisons=comparisons,
        metrics=metrics_data
    )

if __name__ == "__main__":
    # Start Flask Web App without auto-reloader to prevent process restarts during image processing
    print("[*] Starting Heritage Restoration Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
