"""
Improvements 5–9 – Combined Backend Modules
============================================
Each section is a self-contained module.  Import what you need in app.py.

  5. attention_unet.py  – AttentionUNet architecture function
  6. tissue_classifier  – Thin CNN for granulation/slough/necrotic/epithelial
  7. patient_db         – SQLite patient session store + Flask blueprint
  8. infection_score    – Erythema ring + exudate colour infection risk scoring
  9. healing_forecast   – Linear + exponential healing velocity model
"""

# ══════════════════════════════════════════════════════════════════════════════
# 5. ATTENTION U-NET  (replaces build_unet_256 if attention is wanted standalone)
# ══════════════════════════════════════════════════════════════════════════════
"""
attention_unet.py
------------------
Standalone attention-gate U-Net.  Already embedded in retrain_256.py;
this file exposes it as an importable function for other scripts.

from attention_unet import build_attention_unet
model = build_attention_unet(input_shape=(256,256,3))
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


def _conv_block(x, f):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def _attention_gate(x, g, f):
    """
    Soft attention gate (Oktay et al., 2018).
    x = skip connection feature map
    g = decoder gate signal (coarser scale)
    """
    tx = layers.Conv2D(f, 1, padding="same")(x)
    tg = layers.Conv2D(f, 1, padding="same")(g)
    # upsample g to match x if needed
    if tx.shape[1] != tg.shape[1]:
        scale = tx.shape[1] // tg.shape[1]
        tg = layers.UpSampling2D(size=(scale, scale))(tg)
    add = layers.Activation("relu")(layers.Add()([tx, tg]))
    psi = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(add)
    return layers.Multiply()([x, psi])


def _decoder_block(x, skip, f):
    x  = layers.Conv2DTranspose(f, 3, strides=2, padding="same")(x)
    sk = _attention_gate(skip, x, f // 2)
    x  = layers.Concatenate()([x, sk])
    return _conv_block(x, f)


def build_attention_unet(input_shape=(256, 256, 3),
                          pretrained_encoder_path=None):
    """
    Parameters
    ----------
    input_shape              : (H, W, 3) – supports any multiple of 32
    pretrained_encoder_path  : optional path to a saved model from which
                               ResNet50 layer weights are warm-started
    Returns
    -------
    tf.keras.Model
    """
    inp  = layers.Input(shape=input_shape)
    base = ResNet50(include_top=False, weights="imagenet", input_tensor=inp)

    if pretrained_encoder_path:
        try:
            old = tf.keras.models.load_model(pretrained_encoder_path, compile=False)
            for lyr in base.layers:
                try:
                    base.get_layer(lyr.name).set_weights(
                        old.get_layer(lyr.name).get_weights())
                except (ValueError, AttributeError):
                    pass
        except Exception:
            pass  # fallback to ImageNet init

    s1 = base.get_layer("conv1_relu").output          # H/2
    s2 = base.get_layer("conv2_block3_out").output    # H/4
    s3 = base.get_layer("conv3_block4_out").output    # H/8
    s4 = base.get_layer("conv4_block6_out").output    # H/16
    bn = base.get_layer("conv5_block3_out").output    # H/32

    d1 = _decoder_block(bn, s4, 512)
    d2 = _decoder_block(d1, s3, 256)
    d3 = _decoder_block(d2, s2, 128)
    d4 = _decoder_block(d3, s1, 64)
    x  = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(d4)
    x  = _conv_block(x, 32)
    out = layers.Conv2D(1, 1, activation="sigmoid", name="mask_output")(x)

    return Model(inp, out, name=f"attention_unet_{input_shape[0]}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TISSUE CLASSIFIER  (lightweight CNN replacing the HSV heuristic)
# ══════════════════════════════════════════════════════════════════════════════
"""
tissue_classifier.py
---------------------
Train a small CNN head on top of MobileNetV2 to classify wound tissue patches.

Labels (4-class):
  0 = granulation   1 = slough   2 = necrotic   3 = epithelial

Training data layout:
  data/tissue/train/granulation/*.jpg
  data/tissue/train/slough/*.jpg
  data/tissue/train/necrotic/*.jpg
  data/tissue/train/epithelial/*.jpg
  data/tissue/val/...

Usage
-----
  python tissue_classifier.py --train  data/tissue  --output  models/tissue.keras
  # then in app.py:
  from tissue_classifier import classify_tissue_cnn
  tissue = classify_tissue_cnn(img_rgb_01, mask)
"""

import numpy as np
import cv2
from pathlib import Path
import logging

_tissue_log = logging.getLogger("tissue_classifier")
_TISSUE_MODEL = None
TISSUE_LABELS = ["granulation", "slough", "necrotic", "epithelial"]
TISSUE_PATCH_SIZE = 64  # pixels – patch extracted from wound centre


def load_tissue_model(model_path: str):
    global _TISSUE_MODEL
    import tensorflow as tf
    try:
        _TISSUE_MODEL = tf.keras.models.load_model(model_path, compile=False)
        _tissue_log.info("Tissue classifier loaded from %s", model_path)
    except Exception as e:
        _tissue_log.warning("Could not load tissue model: %s – falling back to HSV", e)


def classify_tissue_cnn(img_rgb_01: np.ndarray,
                         mask: np.ndarray,
                         model_path: str = "models/tissue.keras") -> dict:
    """
    If the trained CNN is available, use it.
    Falls back to HSV heuristic (same as v3 app.py) if not.
    """
    global _TISSUE_MODEL
    if _TISSUE_MODEL is None and Path(model_path).exists():
        load_tissue_model(model_path)

    if _TISSUE_MODEL is None:
        return _tissue_hsv_fallback(img_rgb_01, mask)

    return _tissue_cnn_predict(img_rgb_01, mask)


def _tissue_cnn_predict(img_rgb_01, mask):
    import tensorflow as tf
    img_u8 = (np.clip(img_rgb_01, 0, 1) * 255).astype(np.uint8)

    # Extract patches inside the wound region
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return {l: 0.0 for l in TISSUE_LABELS}

    # Sample up to 64 patches
    step    = max(1, len(ys) // 64)
    patches = []
    h, w    = img_u8.shape[:2]
    half    = TISSUE_PATCH_SIZE // 2

    for y, x in zip(ys[::step], xs[::step]):
        y1, y2 = max(0, y - half), min(h, y + half)
        x1, x2 = max(0, x - half), min(w, x + half)
        patch = img_u8[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        patch = cv2.resize(patch, (TISSUE_PATCH_SIZE, TISSUE_PATCH_SIZE))
        patches.append(patch.astype(np.float32) / 255.0)

    if not patches:
        return {l: 0.0 for l in TISSUE_LABELS}

    preds   = _TISSUE_MODEL.predict(np.array(patches), verbose=0)  # (N, 4)
    avg     = preds.mean(axis=0)
    total   = avg.sum() + 1e-8
    result  = {TISSUE_LABELS[i]: round(float(avg[i] / total * 100), 1)
               for i in range(4)}
    result["method"] = "CNN (trained classifier)"
    return result


def _tissue_hsv_fallback(img_rgb_01, mask):
    """Same HSV heuristic as in app.py v3."""
    img_u8  = (np.clip(img_rgb_01, 0, 1) * 255).astype(np.uint8)
    hsv     = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    wound   = mask > 0
    H       = hsv[:, :, 0][wound].astype(np.float32)
    S       = hsv[:, :, 1][wound].astype(np.float32)
    V       = hsv[:, :, 2][wound].astype(np.float32)
    n       = len(H)
    if n == 0:
        return {l: 0.0 for l in TISSUE_LABELS}

    counts = np.array([
        np.sum(((H >= 340) | (H <= 15)) & (S > 80)),
        np.sum((H >= 40) & (H <= 80) & (S > 40) & (V > 60)),
        np.sum(V < 60),
        np.sum((S < 40) & (V > 150)),
    ], dtype=float)
    other  = max(0.0, n - counts.sum())
    pcts   = np.round(np.append(counts, other) / n * 100, 1).tolist()
    return {
        "granulation": pcts[0], "slough": pcts[1],
        "necrotic": pcts[2],    "epithelial": pcts[3],
        "other": pcts[4],       "method": "HSV heuristic (no trained model found)",
    }


def build_tissue_cnn():
    """Build the lightweight MobileNetV2 tissue classifier for training."""
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(
        input_shape=(TISSUE_PATCH_SIZE, TISSUE_PATCH_SIZE, 3),
        include_top=False, weights="imagenet")
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(base.input, out, name="tissue_cnn")


# CLI training entry-point
if __name__ == "__main__":
    import argparse, tensorflow as tf
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",  required=True, help="path to data/tissue folder")
    ap.add_argument("--output", default="models/tissue.keras")
    args = ap.parse_args()

    dg = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, rotation_range=30, horizontal_flip=True,
        zoom_range=0.2, validation_split=0.2)
    train_gen = dg.flow_from_directory(
        f"{args.train}/train",
        target_size=(TISSUE_PATCH_SIZE, TISSUE_PATCH_SIZE),
        batch_size=32, class_mode="categorical")
    val_gen = dg.flow_from_directory(
        f"{args.train}/val",
        target_size=(TISSUE_PATCH_SIZE, TISSUE_PATCH_SIZE),
        batch_size=32, class_mode="categorical")

    model = build_tissue_cnn()
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=30,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(
                  args.output, save_best_only=True, monitor="val_accuracy")])
    print(f"Saved to {args.output}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PATIENT DATABASE  (SQLite + Flask blueprint)
# ══════════════════════════════════════════════════════════════════════════════
"""
patient_db.py
--------------
Lightweight SQLite session store.  In production swap for PostgreSQL
by changing DB_URL.

Register in app.py:
  from patient_db import db_bp, init_db
  init_db()
  app.register_blueprint(db_bp)

Endpoints added:
  POST /patients                 – create patient record
  GET  /patients/<id>            – get patient + scan history
  POST /patients/<id>/scans      – save a scan result for a patient
  GET  /patients/<id>/scans      – list all scans for patient
  GET  /patients/<id>/scans/compare?scan_a=<id>&scan_b=<id>
"""

import sqlite3, json, os
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g

DB_PATH = os.getenv("DB_PATH", "woundai.db")
db_bp   = Blueprint("patient_db", __name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    dob         TEXT,
    notes       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
    id          TEXT PRIMARY KEY,
    patient_id  TEXT NOT NULL REFERENCES patients(id),
    scan_date   TEXT NOT NULL,
    metrics     TEXT,          -- JSON
    skin        TEXT,          -- JSON
    healing     TEXT,          -- JSON
    tissue      TEXT,          -- JSON
    report      TEXT,          -- JSON
    mask_b64    TEXT,
    heatmap_b64 TEXT,
    overlay_b64 TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scans_patient ON scans(patient_id);
CREATE INDEX IF NOT EXISTS idx_scans_date    ON scans(scan_date);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


@db_bp.teardown_app_request
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()


@db_bp.route("/patients", methods=["POST"])
def create_patient():
    data = request.get_json() or {}
    import uuid
    pid = str(uuid.uuid4())
    db  = get_db()
    db.execute(
        "INSERT INTO patients VALUES (?,?,?,?,?)",
        [pid, data.get("name",""), data.get("dob",""),
         data.get("notes",""), datetime.now(timezone.utc).isoformat()])
    db.commit()
    return jsonify({"id": pid}), 201


@db_bp.route("/patients/<pid>", methods=["GET"])
def get_patient(pid):
    db  = get_db()
    row = db.execute("SELECT * FROM patients WHERE id=?", [pid]).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    scans = db.execute(
        "SELECT id,scan_date,created_at FROM scans WHERE patient_id=? ORDER BY scan_date DESC",
        [pid]).fetchall()
    return jsonify({**dict(row), "scans": [dict(s) for s in scans]})


@db_bp.route("/patients/<pid>/scans", methods=["POST"])
def save_scan(pid):
    data = request.get_json() or {}
    import uuid
    sid = str(uuid.uuid4())
    db  = get_db()
    db.execute(
        """INSERT INTO scans
           (id,patient_id,scan_date,metrics,skin,healing,tissue,report,
            mask_b64,heatmap_b64,overlay_b64,created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [sid, pid,
         data.get("scan_date", datetime.now(timezone.utc).date().isoformat()),
         json.dumps(data.get("metrics",  {})),
         json.dumps(data.get("skin",     {})),
         json.dumps(data.get("healing",  {})),
         json.dumps(data.get("tissue",   {})),
         json.dumps(data.get("report",   {})),
         data.get("mask_image",    ""),
         data.get("heatmap_image", ""),
         data.get("overlay_image", ""),
         datetime.now(timezone.utc).isoformat()])
    db.commit()
    return jsonify({"scan_id": sid}), 201


@db_bp.route("/patients/<pid>/scans", methods=["GET"])
def list_scans(pid):
    db   = get_db()
    rows = db.execute(
        "SELECT * FROM scans WHERE patient_id=? ORDER BY scan_date DESC", [pid]).fetchall()
    return jsonify([{
        **dict(r),
        "metrics": json.loads(r["metrics"] or "{}"),
        "healing": json.loads(r["healing"] or "{}"),
    } for r in rows])


# ══════════════════════════════════════════════════════════════════════════════
# 8. INFECTION RISK SCORE
# ══════════════════════════════════════════════════════════════════════════════
"""
infection_score.py
-------------------
Rule-based infection risk scoring using:
  • Erythema ring  – red halo outside the wound contour
  • Exudate colour – yellow/green tint inside wound (slough proxy)
  • Odour proxy    – not computable from image; placeholder for future EHR input

Returns:
  {
    "risk_level":  "low" | "moderate" | "high",
    "risk_score":  0.0–1.0,
    "erythema_pct": float,   # % of peri-wound pixels showing erythema
    "exudate_score": float,  # 0–1
    "flags": [str]
  }
"""


def score_infection(img_rgb_01: np.ndarray, mask: np.ndarray,
                    peri_margin_px: int = 20) -> dict:
    """
    Parameters
    ----------
    img_rgb_01     : float32 (H,W,3) normalised image
    mask           : uint8 (H,W) binary wound mask {0,255}
    peri_margin_px : width of peri-wound ring to check for erythema
    """
    img_u8 = (np.clip(img_rgb_01, 0, 1) * 255).astype(np.uint8)
    hsv    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)

    # Peri-wound ring: dilate mask and subtract original
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (peri_margin_px*2+1, peri_margin_px*2+1))
    dilated = cv2.dilate(mask, kernel)
    peri    = cv2.subtract(dilated, mask)          # ring only

    # Erythema: red hue (0-10° or 160-180° in HSV) + medium saturation
    peri_bool = peri > 0
    H_peri    = hsv[:, :, 0][peri_bool]
    S_peri    = hsv[:, :, 1][peri_bool]
    erythema_px = np.sum(((H_peri <= 10) | (H_peri >= 160)) & (S_peri > 60))
    erythema_pct = float(erythema_px / max(peri_bool.sum(), 1) * 100)

    # Exudate: yellow/green inside wound (H 30–90, S > 50, V > 50)
    wound_bool   = mask > 0
    H_wound      = hsv[:, :, 0][wound_bool]
    S_wound      = hsv[:, :, 1][wound_bool]
    V_wound      = hsv[:, :, 2][wound_bool]
    exudate_px   = np.sum((H_wound >= 30) & (H_wound <= 90) &
                          (S_wound > 50) & (V_wound > 50))
    exudate_score = float(exudate_px / max(wound_bool.sum(), 1))

    # Composite risk
    risk_score = min(1.0, erythema_pct / 40.0 * 0.6 + exudate_score * 0.4)
    flags = []
    if erythema_pct > 20: flags.append(f"Erythema ring detected ({erythema_pct:.1f}% of peri-wound)")
    if exudate_score > 0.25: flags.append(f"Possible exudate/slough ({exudate_score:.0%} of wound)")

    if   risk_score >= 0.55: risk_level = "high"
    elif risk_score >= 0.25: risk_level = "moderate"
    else:                    risk_level = "low"

    return {
        "risk_level":    risk_level,
        "risk_score":    round(risk_score, 3),
        "erythema_pct":  round(erythema_pct, 1),
        "exudate_score": round(exudate_score, 3),
        "flags":         flags,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. HEALING FORECAST  (linear + exponential)
# ══════════════════════════════════════════════════════════════════════════════
"""
healing_forecast.py
--------------------
Given a list of (date, area_pct) tuples from past scans, fits a model and
returns predicted closure date and trajectory.

from healing_forecast import forecast_closure
scans = [("2026-01-01", 8.5), ("2026-01-08", 6.2), ("2026-01-15", 4.8)]
result = forecast_closure(scans)
"""

from datetime import datetime, timedelta, timezone


def forecast_closure(scans: list, method: str = "auto") -> dict:
    """
    Parameters
    ----------
    scans  : list of (date_str, area_pct) – chronological order, at least 2
    method : "linear" | "exponential" | "auto"
              auto = exponential if R² > 0.85, else linear

    Returns
    -------
    {
      "method": str,
      "r_squared": float,
      "healing_rate_per_day": float,   # negative = improving
      "projected_closure_date": str,   # ISO date
      "days_to_closure": int,
      "trajectory": [{"day": int, "area_pct": float}, ...]
    }
    """
    if len(scans) < 2:
        return {"error": "Need at least 2 scans to forecast"}

    def to_day(s):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try: return datetime.strptime(s, fmt)
            except: pass
        return datetime.now(timezone.utc)

    dates    = [to_day(s[0]) for s in scans]
    areas    = [float(s[1]) for s in scans]
    day0     = dates[0]
    days_arr = np.array([(d - day0).days for d in dates], dtype=float)
    area_arr = np.array(areas, dtype=float)

    # Linear fit
    A_lin = np.vstack([days_arr, np.ones_like(days_arr)]).T
    m_lin, b_lin = np.linalg.lstsq(A_lin, area_arr, rcond=None)[0]
    pred_lin = m_lin * days_arr + b_lin
    ss_res_l = np.sum((area_arr - pred_lin) ** 2)
    ss_tot   = np.sum((area_arr - area_arr.mean()) ** 2) + 1e-12
    r2_lin   = 1 - ss_res_l / ss_tot

    # Exponential fit: area = a * exp(b * day)
    try:
        log_a   = np.log(np.clip(area_arr, 1e-3, None))
        A_exp   = np.vstack([days_arr, np.ones_like(days_arr)]).T
        b_e, ln_a = np.linalg.lstsq(A_exp, log_a, rcond=None)[0]
        a_e     = np.exp(ln_a)
        pred_exp = a_e * np.exp(b_e * days_arr)
        ss_res_e = np.sum((area_arr - pred_exp) ** 2)
        r2_exp   = 1 - ss_res_e / ss_tot
    except Exception:
        r2_exp, b_e, a_e = 0.0, m_lin, area_arr[0]

    # Choose method
    if method == "auto":
        use_exp = r2_exp > 0.85 and r2_exp > r2_lin
    else:
        use_exp = method == "exponential"

    last_day  = days_arr[-1]
    last_area = area_arr[-1]

    if use_exp:
        label = "exponential"
        r2    = round(r2_exp, 3)
        # Days until exp reaches ~0.1 %
        if b_e < 0:
            days_close = int((math.log(0.1 / a_e) / b_e) - last_day)
        else:
            days_close = None
        rate = round(float(b_e * last_area), 4)  # instantaneous rate
    else:
        label = "linear"
        r2    = round(r2_lin, 3)
        days_close = int(-last_area / m_lin) if m_lin < 0 else None
        rate  = round(float(m_lin), 4)

    days_close = max(0, days_close) if days_close is not None else None
    closure_date = (datetime.now(timezone.utc) + timedelta(days=days_close)).date().isoformat() \
                   if days_close is not None else None

    # Trajectory: next 60 days
    traj_days = list(range(0, min(90, (days_close or 60) + 10), 7))
    if use_exp:
        traj = [{"day": d, "area_pct": round(float(a_e * np.exp(b_e * (last_day + d))), 2)}
                for d in traj_days]
    else:
        traj = [{"day": d, "area_pct": round(float(max(0, m_lin * (last_day + d) + b_lin)), 2)}
                for d in traj_days]

    return {
        "method":                   label,
        "r_squared":                r2,
        "healing_rate_per_day":     rate,
        "projected_closure_date":   closure_date,
        "days_to_closure":          days_close,
        "trajectory":               traj,
    }
