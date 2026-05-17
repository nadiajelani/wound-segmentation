"""
Integration Patch – Wire All 9 Improvements into app.py
=========================================================
Apply each PATCH block in order.  Each one shows the exact existing lines
to find and what to replace them with (or insert after).

Run order
---------
  PATCH 0 – Add new imports at top of app.py
  PATCH 1 – Update IMG_SIZE constant
  PATCH 2 – Call quality gate before inference
  PATCH 3 – Call calibration after preprocessing
  PATCH 4 – Replace tissue heuristic with CNN wrapper
  PATCH 5 – Add infection score to analyze result
  PATCH 6 – Add healing forecast to analyze result
  PATCH 7 – Register blueprints (progress + patient DB)
  PATCH 8 – Update /version endpoint to reflect new capabilities
"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 0 – New imports  (add after existing imports at top of app.py)
# ─────────────────────────────────────────────────────────────────────────────
PATCH_0_INSERT_AFTER = "import tensorflow as tf"
PATCH_0_NEW_LINES = """
# ── Improvement modules ──────────────────────────────────────────────────────
try:
    from image_quality    import check_quality, QualityError
    QUALITY_GATE_ENABLED = True
except ImportError:
    QUALITY_GATE_ENABLED = False

try:
    from calibration      import calibrate_image, apply_calibration
    CALIBRATION_ENABLED  = True
except ImportError:
    CALIBRATION_ENABLED  = False

try:
    from tissue_classifier import classify_tissue_cnn
    TISSUE_CNN_ENABLED   = True
except ImportError:
    TISSUE_CNN_ENABLED   = False

try:
    from infection_score  import score_infection
    INFECTION_ENABLED    = True
except ImportError:
    INFECTION_ENABLED    = False

try:
    from healing_forecast import forecast_closure
    FORECAST_ENABLED     = True
except ImportError:
    FORECAST_ENABLED     = False

try:
    from progress_tracker import compare_wounds
    from patient_db       import db_bp, init_db
    DB_ENABLED           = True
except ImportError:
    DB_ENABLED           = False
"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 – Update IMG_SIZE  (change the constant)
# ─────────────────────────────────────────────────────────────────────────────
PATCH_1_FIND    = 'IMG_SIZE = (int(os.getenv("IMG_H", 128)), int(os.getenv("IMG_W", 128)))'
PATCH_1_REPLACE = 'IMG_SIZE = (int(os.getenv("IMG_H", 256)), int(os.getenv("IMG_W", 256)))'
# NOTE: set IMG_H=128 IMG_W=128 in your env if you still use the old model.
#       When the new 256-model is deployed, leave as default 256.

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 – Quality gate  (insert after preprocess_image() call in analyze_wound)
# ─────────────────────────────────────────────────────────────────────────────
PATCH_2_FIND = """        # ── 2. Preprocess ───────────────────────────────────────────────────
        try:
            img_arr = preprocess_image(raw)          # (1, H, W, 3)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 422"""

PATCH_2_REPLACE = """        # ── 2. Preprocess ───────────────────────────────────────────────────
        try:
            img_arr = preprocess_image(raw)          # (1, H, W, 3)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 422

        # ── 2b. Quality gate ────────────────────────────────────────────────
        if QUALITY_GATE_ENABLED:
            try:
                q_report = check_quality(img_arr[0], raise_on_fail=True)
            except QualityError as qe:
                return jsonify({"error": str(qe),
                                "quality_report": qe.report,
                                "code": "QUALITY_FAIL"}), 422"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 – Calibration  (insert after calculate_metrics() call)
# ─────────────────────────────────────────────────────────────────────────────
PATCH_3_FIND = """        # ── 4. Extended metrics ─────────────────────────────────────────────
        metrics = calculate_metrics(mask, mean_pm, std_pm)"""

PATCH_3_REPLACE = """        # ── 4. Extended metrics ─────────────────────────────────────────────
        metrics = calculate_metrics(mask, mean_pm, std_pm)

        # ── 4b. Real-world calibration (mm²) ────────────────────────────────
        if CALIBRATION_ENABLED:
            cal = calibrate_image(img_arr[0], method="auto")
            metrics = apply_calibration(metrics, cal)
        else:
            metrics["area_mm2"]    = None
            metrics["perimeter_mm"]= None"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 4 – Replace tissue heuristic with CNN wrapper
# ─────────────────────────────────────────────────────────────────────────────
PATCH_4_FIND    = "        tissue = classify_tissue_types(img_arr[0], mask)"
PATCH_4_REPLACE = """        # Use CNN classifier if available, else HSV fallback
        if TISSUE_CNN_ENABLED:
            tissue = classify_tissue_cnn(img_arr[0], mask)
        else:
            tissue = classify_tissue_types(img_arr[0], mask)"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 5 – Infection score  (add to result dict before return)
# ─────────────────────────────────────────────────────────────────────────────
PATCH_5_FIND = """        result = {
            "success":          True,
            "version":          VERSION,"""

PATCH_5_REPLACE = """        # ── Infection risk ──────────────────────────────────────────────────
        infection = {}
        if INFECTION_ENABLED:
            infection = score_infection(img_arr[0], mask)

        result = {
            "success":          True,
            "version":          VERSION,
            "infection_risk":   infection,"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 6 – Register blueprints  (add after app = Flask(__name__))
# ─────────────────────────────────────────────────────────────────────────────
PATCH_6_FIND    = "app = Flask(__name__)"
PATCH_6_REPLACE = """app = Flask(__name__)

if DB_ENABLED:
    init_db()
    app.register_blueprint(db_bp)
    app.register_blueprint(compare_wounds)
    logger.info("Patient DB and progress tracker enabled")"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 7 – /version  update
# ─────────────────────────────────────────────────────────────────────────────
PATCH_7_FIND = """    return jsonify({
        "version":          VERSION,
        "model_path":       MODEL_PATH,
        "img_size":         list(IMG_SIZE),
        "mc_passes":        MC_PASSES,
        "skin_method":      "ITA (CIE L*a*b*)",
        "threshold_method": "Otsu adaptive",
    })"""

PATCH_7_REPLACE = """    return jsonify({
        "version":              VERSION,
        "model_path":           MODEL_PATH,
        "img_size":             list(IMG_SIZE),
        "mc_passes":            MC_PASSES,
        "skin_method":          "ITA (CIE L*a*b*)",
        "threshold_method":     "Otsu adaptive",
        "features": {
            "quality_gate":     QUALITY_GATE_ENABLED,
            "calibration_mm2":  CALIBRATION_ENABLED,
            "tissue_cnn":       TISSUE_CNN_ENABLED,
            "infection_score":  INFECTION_ENABLED,
            "healing_forecast": FORECAST_ENABLED,
            "patient_db":       DB_ENABLED,
        }
    })"""

# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT NOTES
# ─────────────────────────────────────────────────────────────────────────────
NOTES = """
Deployment checklist
=====================
1. Copy all improvement module files alongside app.py:
     image_quality.py
     calibration.py
     tissue_classifier.py   (with models/tissue.keras if trained)
     infection_score.py
     healing_forecast.py
     progress_tracker.py
     patient_db.py

2. Set environment variables (Google Cloud Run → Variables):
     IMG_H=256
     IMG_W=256
     MODEL_PATH=models/wound_seg_256.keras
     DB_PATH=/tmp/woundai.db   (or mount a persistent volume)

3. Update requirements.txt to include:
     scipy
     scikit-image

4. If deploying tissue CNN, copy models/tissue.keras to the container
   and set TISSUE_MODEL_PATH env var.

5. For coin calibration to work, instruct users to place a known coin
   (e.g. Australian $1 = 25mm) at the edge of the wound image before
   taking the photo.  Update the UI to show this tip.

6. Test locally:
     python -c "from image_quality import check_quality; import numpy as np; \
       img = np.random.rand(256,256,3).astype('float32'); \
       r = check_quality(img, raise_on_fail=False); print(r.to_dict())"
"""

if __name__ == "__main__":
    print(NOTES)
    print("Apply patches manually to app.py following the PATCH_* dictionaries above.")
    print("Each PATCH has FIND (exact string to locate) and REPLACE (new string).")
