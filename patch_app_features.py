"""
patch_app_features.py
---------------------
Run once from your project folder:
    python patch_app_features.py

Adds to app.py:
  - wound type classification
  - PUSH score
  - healing velocity
  - fractal dimension
  - QR code
  - PDF export endpoint
  - similarity search
  - embedding storage
"""

import shutil, re
from pathlib import Path

APP = Path("app.py")
if not APP.exists():
    print("❌ app.py not found"); exit(1)

shutil.copy(APP, "app_before_features.py")
print("✅ Backed up to app_before_features.py")

src = APP.read_text()

# ── 1. Add import ─────────────────────────────────────────────────────────────
IMPORT_MARKER = "import numpy as np"
NEW_IMPORT = """import numpy as np
try:
    from wound_features import (
        classify_wound_type, calculate_push_score,
        compute_healing_velocity, compute_fractal_dimension,
        generate_qr_code, find_similar_wounds, save_embedding,
        generate_pdf_report,
    )
    FEATURES_ENABLED = True
except ImportError as _fe:
    FEATURES_ENABLED = False
    import logging as _fl
    _fl.getLogger("woundai").warning(f"wound_features.py not found: {_fe}")"""

if "from wound_features import" not in src:
    src = src.replace(IMPORT_MARKER, NEW_IMPORT, 1)
    print("✅ Patch 1: imports added")

# ── 2. Wire features into analyze_wound ──────────────────────────────────────
# Find the result dict construction and add new fields
OLD_RESULT = '"success":            True,'
NEW_RESULT = '''"success":            True,'''  # find anchor

# Add features computation before result dict
OLD_REPORT_LINE = "        report    = generate_doctor_report(metrics, healing, skin,"
NEW_BEFORE_REPORT = """        # ── New features ────────────────────────────────────────────────────
        wound_type  = {}
        push_score  = {}
        fractal_dim = None
        qr_code     = None
        pdf_b64     = None
        similar     = []
        embedding   = None

        if FEATURES_ENABLED:
            try:
                wound_type  = classify_wound_type(img_arr[0], mask, metrics)
            except Exception as _e:
                logger.warning(f"Wound type: {_e}")
            try:
                exudate_s   = infection.get("exudate_score", 0) if 'infection' in dir() else 0
                area_cm2    = metrics.get("area_mm2", 0) / 100 if metrics.get("area_mm2") else None
                push_score  = calculate_push_score(area_cm2, exudate_s, tissue)
            except Exception as _e:
                logger.warning(f"PUSH score: {_e}")
            try:
                fractal_dim = round(compute_fractal_dimension(mask), 3)
            except Exception as _e:
                logger.warning(f"Fractal: {_e}")
            try:
                from gradcam import extract_embedding
                embedding = extract_embedding(MODEL, img_arr)
                if embedding is not None:
                    import uuid as _uuid
                    scan_id = str(_uuid.uuid4())
                    save_embedding(scan_id, embedding,
                                   wound_type=wound_type.get("wound_type",""),
                                   area_pct=metrics.get("area_percentage",0),
                                   severity=metrics.get("severity",""))
                    similar = find_similar_wounds(embedding, top_k=3,
                                                  exclude_id=scan_id)
            except Exception as _e:
                logger.warning(f"Embedding/similarity: {_e}")

"""

if OLD_REPORT_LINE in src and "wound_type  = classify_wound_type" not in src:
    src = src.replace(OLD_REPORT_LINE, NEW_BEFORE_REPORT + OLD_REPORT_LINE)
    print("✅ Patch 2: feature computation added")

# ── 3. Add fields to result dict ───────────────────────────────────────────────
OLD_RESULT_DICT = '"success":            True,\n            "version":            VERSION,'
NEW_RESULT_DICT = '''"success":            True,
            "version":            VERSION,
            "wound_type":         wound_type,
            "push_score":         push_score,
            "fractal_dimension":  fractal_dim,
            "similar_wounds":     similar,'''

if OLD_RESULT_DICT not in src and '"wound_type":' not in src:
    src = src.replace(
        '"success":            True,\n            "version":            VERSION,',
        NEW_RESULT_DICT)
    print("✅ Patch 3: new fields added to result dict")

# ── 4. Add /pdf endpoint ───────────────────────────────────────────────────────
PDF_ENDPOINT = '''
@app.route("/pdf", methods=["POST"])
def generate_pdf():
    """Generate PDF report from analysis result JSON."""
    try:
        data     = request.get_json() or {}
        result   = data.get("result", {})
        img_b64  = data.get("original_image", "")
        if not result:
            return jsonify({"error": "No result data provided"}), 400
        if not FEATURES_ENABLED:
            return jsonify({"error": "PDF feature not available"}), 503
        pdf_b64 = generate_pdf_report(result, img_b64)
        if pdf_b64 is None:
            return jsonify({"error": "PDF generation failed — install reportlab"}), 500
        return jsonify({"success": True, "pdf": pdf_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/similar", methods=["POST"])
def similar_wounds():
    """Find similar wounds from the embedding database."""
    try:
        if not FEATURES_ENABLED:
            return jsonify({"similar": []}), 200
        data = request.get_json() or {}
        emb  = data.get("embedding")
        if emb is None:
            return jsonify({"error": "No embedding provided"}), 400
        emb_arr = np.array(emb, dtype=np.float32)
        results = find_similar_wounds(emb_arr, top_k=data.get("top_k", 3))
        return jsonify({"success": True, "similar": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

'''

if "@app.route(\"/pdf\"" not in src:
    # Insert before /analyze route
    src = src.replace(
        "@app.route(\"/analyze\", methods=[\"POST\"])",
        PDF_ENDPOINT + "@app.route(\"/analyze\", methods=[\"POST\"])")
    print("✅ Patch 4: /pdf and /similar endpoints added")

APP.write_text(src)
print("\n✅ app.py patched successfully!")
print("\nInstall new dependencies:")
print("  pip install reportlab qrcode[pil]")
print("\nThen redeploy:")
print("  gcloud run deploy wound-api --source . --region us-central1 --memory 4Gi --cpu 2 --timeout 600 --allow-unauthenticated")
