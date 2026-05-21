"""
wound_features.py
==================
Drop next to app.py and import everything from here.

Import in app.py:
    from wound_features import (
        classify_wound_type,
        calculate_push_score,
        generate_pdf_report,
        compute_healing_velocity,
        compute_fractal_dimension,
        generate_qr_code,
        find_similar_wounds,
        save_embedding,
    )
"""

import os, io, math, json, base64, logging, hashlib, sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import cv2

log = logging.getLogger("woundai.features")

# ─────────────────────────────────────────────────────────────────────────────
# 1. WOUND TYPE CLASSIFICATION
#    Rule-based using colour + shape + location heuristics.
#    Replace body with wound_type_clf.keras when available.
# ─────────────────────────────────────────────────────────────────────────────

WOUND_TYPES = {
    "diabetic_foot": {
        "label":       "Diabetic Foot Ulcer",
        "short":       "DFU",
        "description": "Neuropathic or ischaemic ulceration typically on plantar surface.",
        "recommendations": [
            "Urgent vascular assessment (ABPI)",
            "HbA1c optimisation — target <7%",
            "Podiatrist referral within 24 hours",
            "Total contact casting if neuropathic",
            "Offloading footwear essential",
            "Monitor for osteomyelitis (X-ray or MRI)",
        ],
        "icd10": "E11.621",
    },
    "pressure_ulcer": {
        "label":       "Pressure Ulcer / Injury",
        "short":       "PU",
        "description": "Localised injury to skin and underlying tissue from sustained pressure.",
        "recommendations": [
            "Offload pressure immediately",
            "Calculate PUSH score for monitoring",
            "Foam or air mattress if Stage II+",
            "Nutritional assessment (protein, zinc, Vit C)",
            "Reposition every 2 hours",
            "Tissue viability nurse referral",
        ],
        "icd10": "L89.90",
    },
    "venous_ulcer": {
        "label":       "Venous Leg Ulcer",
        "short":       "VLU",
        "description": "Chronic ulceration of the lower leg due to venous hypertension.",
        "recommendations": [
            "Four-layer compression bandaging",
            "Leg elevation >30° when resting",
            "Exclude arterial disease before compression (ABPI >0.8)",
            "Zinc paste bandage under compression",
            "Address underlying venous reflux",
            "Target healing in 12 weeks; refer if not",
        ],
        "icd10": "I83.009",
    },
    "surgical": {
        "label":       "Surgical / Post-Operative Wound",
        "short":       "SW",
        "description": "Wound resulting from surgical incision or procedure.",
        "recommendations": [
            "Monitor for dehiscence and signs of SSI",
            "Negative pressure wound therapy if dehisced",
            "Maintain aseptic technique at dressing changes",
            "Assess haematoma or seroma",
            "Suture/staple removal per surgical protocol",
            "Document in surgical notes",
        ],
        "icd10": "T81.30",
    },
    "burn": {
        "label":       "Burn Wound",
        "short":       "BURN",
        "description": "Thermal, chemical, or electrical tissue injury.",
        "recommendations": [
            "Cool with running water 20 minutes (not ice)",
            "Assess burn depth and surface area (Rule of Nines)",
            "Silver sulfadiazine or Mepilex Ag dressing",
            "Fluid resuscitation if >15% TBSA (Parkland formula)",
            "Burns unit referral if >10% or face/hands/genitals",
            "Tetanus prophylaxis",
        ],
        "icd10": "T30.0",
    },
    "traumatic": {
        "label":       "Traumatic / Abrasion",
        "short":       "TRAUMA",
        "description": "Mechanical injury — laceration, abrasion, or crush wound.",
        "recommendations": [
            "Irrigate with saline to remove debris",
            "Assess depth and need for closure",
            "Tetanus status check",
            "Antibiotic prophylaxis if contaminated",
            "Document mechanism of injury",
            "Follow-up in 48–72 hours",
        ],
        "icd10": "T14.0",
    },
}


def classify_wound_type(
    img_rgb_01: np.ndarray,
    mask: np.ndarray,
    metrics: dict,
    model_path: str = "models/wound_type_clf.keras",
) -> dict:
    """
    Classify wound type using trained CNN if available,
    otherwise fall back to colour + shape heuristics.
    """
    # Try CNN first
    if Path(model_path).exists():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)
            img_256 = cv2.resize(
                (np.clip(img_rgb_01, 0, 1) * 255).astype(np.uint8),
                (256, 256))
            inp   = np.expand_dims(img_256.astype(np.float32) / 255.0, 0)
            pred  = model.predict(inp, verbose=0)[0]
            # Map to type keys (order matches training)
            keys  = ["diabetic_foot", "pressure_ulcer", "venous_ulcer",
                     "surgical", "traumatic"]
            top_i = int(np.argmax(pred))
            key   = keys[min(top_i, len(keys)-1)]
            conf  = float(pred[top_i])
            info  = WOUND_TYPES[key]
            return {
                "wound_type":    key,
                "label":         info["label"],
                "short":         info["short"],
                "confidence":    round(conf, 3),
                "description":   info["description"],
                "recommendations": info["recommendations"],
                "icd10":         info["icd10"],
                "method":        "CNN classifier",
            }
        except Exception as e:
            log.warning(f"CNN classifier failed: {e} — using heuristic")

    # Heuristic fallback
    return _heuristic_wound_type(img_rgb_01, mask, metrics)


def _heuristic_wound_type(img_rgb_01, mask, metrics) -> dict:
    """Colour + shape heuristic wound type classifier."""
    img_u8  = (np.clip(img_rgb_01, 0, 1) * 255).astype(np.uint8)
    hsv     = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    wound   = mask > 0

    H = hsv[:, :, 0][wound].astype(float)
    S = hsv[:, :, 1][wound].astype(float)
    V = hsv[:, :, 2][wound].astype(float)
    n = max(len(H), 1)

    area_pct  = metrics.get("area_percentage", 0)
    circ      = metrics.get("circularity", 0.5)

    # Feature scores
    yellow_pct  = np.sum((H >= 20) & (H <= 50) & (S > 40) & (V > 80)) / n
    dark_pct    = np.sum(V < 60) / n
    red_pct     = np.sum(((H <= 15) | (H >= 160)) & (S > 60)) / n
    irregular   = circ < 0.5

    scores = {
        "diabetic_foot":  0.0,
        "pressure_ulcer": 0.0,
        "venous_ulcer":   0.0,
        "surgical":       0.0,
        "traumatic":      0.0,
    }

    # Pressure ulcer: circular, moderate size
    if circ > 0.55:      scores["pressure_ulcer"] += 0.4
    if area_pct > 2:     scores["pressure_ulcer"] += 0.2
    if dark_pct > 0.15:  scores["pressure_ulcer"] += 0.2

    # DFU: irregular, yellow/necrotic
    if irregular:        scores["diabetic_foot"]  += 0.3
    if yellow_pct > 0.2: scores["diabetic_foot"]  += 0.3
    if dark_pct > 0.2:   scores["diabetic_foot"]  += 0.2

    # Venous: large, irregular, yellow/fibrinous
    if area_pct > 5:     scores["venous_ulcer"]   += 0.3
    if yellow_pct > 0.3: scores["venous_ulcer"]   += 0.3
    if irregular:        scores["venous_ulcer"]   += 0.2

    # Surgical: regular, linear
    if circ > 0.3 and circ < 0.65: scores["surgical"] += 0.3
    if red_pct > 0.4:    scores["surgical"]        += 0.2

    # Traumatic: high red, any shape
    if red_pct > 0.5:    scores["traumatic"]       += 0.4
    if area_pct < 3:     scores["traumatic"]        += 0.2

    key  = max(scores, key=scores.get)
    conf = min(0.75, max(0.4, scores[key]))
    info = WOUND_TYPES[key]

    return {
        "wound_type":      key,
        "label":           info["label"],
        "short":           info["short"],
        "confidence":      round(conf, 3),
        "description":     info["description"],
        "recommendations": info["recommendations"],
        "icd10":           info["icd10"],
        "method":          "heuristic",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. PUSH SCORE  (Pressure Ulcer Scale for Healing)
#    Clinical standard used by nurses worldwide.
#    Ref: NPUAP, 1998
# ─────────────────────────────────────────────────────────────────────────────

def calculate_push_score(
    area_cm2: Optional[float],
    exudate_score: float,
    tissue_composition: dict,
) -> dict:
    """
    PUSH Tool 3.0

    Sub-scores
    ----------
    Length × Width (cm²):
        0 = 0, 1 = 0.3–0.6, 2 = 0.7–1.0, 3 = 1.1–2.0, 4 = 2.1–3.0
        5 = 3.1–4.0, 6 = 4.1–8.0, 7 = 8.1–12.0, 8 = 12.1–24.0, 9 = >24.0

    Exudate amount (from 0–1 score):
        0 = none, 1 = light (<0.25), 2 = moderate (0.25–0.5), 3 = heavy (>0.5)

    Tissue type (from composition):
        0 = closed/epithelial, 1 = epithelial, 2 = granulation,
        3 = slough, 4 = necrotic
    """
    # Area sub-score
    if area_cm2 is None or area_cm2 == 0:
        area_sub = 0
    elif area_cm2 <= 0.3:  area_sub = 1
    elif area_cm2 <= 0.6:  area_sub = 1
    elif area_cm2 <= 1.0:  area_sub = 2
    elif area_cm2 <= 2.0:  area_sub = 3
    elif area_cm2 <= 3.0:  area_sub = 4
    elif area_cm2 <= 4.0:  area_sub = 5
    elif area_cm2 <= 8.0:  area_sub = 6
    elif area_cm2 <= 12.0: area_sub = 7
    elif area_cm2 <= 24.0: area_sub = 8
    else:                  area_sub = 9

    # Exudate sub-score
    if   exudate_score < 0.05:  exu_sub = 0
    elif exudate_score < 0.25:  exu_sub = 1
    elif exudate_score < 0.50:  exu_sub = 2
    else:                       exu_sub = 3

    # Tissue sub-score
    necr  = tissue_composition.get("necrotic",    0)
    slou  = tissue_composition.get("slough",       0)
    gran  = tissue_composition.get("granulation",  0)
    epth  = tissue_composition.get("epithelial",   0)

    if   necr  > 20: tis_sub = 4
    elif slou  > 20: tis_sub = 3
    elif gran  > 50: tis_sub = 2
    elif epth  > 50: tis_sub = 1
    else:            tis_sub = 2  # default granulation

    total = area_sub + exu_sub + tis_sub

    # Interpretation
    if   total <= 4:  interpretation = "Healing well — continue current plan"
    elif total <= 9:  interpretation = "Some healing — review at 2 weeks"
    elif total <= 14: interpretation = "Slow healing — consider advanced therapy"
    else:             interpretation = "Not healing — urgent clinical review"

    return {
        "total":              total,
        "max":                17,
        "area_subscore":      area_sub,
        "exudate_subscore":   exu_sub,
        "tissue_subscore":    tis_sub,
        "interpretation":     interpretation,
        "note": "PUSH score: 0=healed, 17=worst. Decrease over time = healing.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. HEALING VELOCITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_healing_velocity(scans: list) -> dict:
    """
    scans: [{"date": "YYYY-MM-DD", "area_pct": float}, ...]
    Returns healing rate, projected closure date, trajectory.
    """
    if len(scans) < 2:
        return {}
    try:
        from datetime import datetime, timedelta
        def to_dt(s):
            for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
                try: return datetime.strptime(s, fmt)
                except: pass
            return datetime.now()

        dates = [to_dt(s["date"]) for s in scans]
        areas = [float(s["area_pct"]) for s in scans]
        d0    = dates[0]
        days  = np.array([(d - d0).days for d in dates], dtype=float)
        arr   = np.array(areas, dtype=float)

        A     = np.vstack([days, np.ones_like(days)]).T
        m, b  = np.linalg.lstsq(A, arr, rcond=None)[0]

        last_day  = float(days[-1])
        last_area = float(arr[-1])

        days_close = int(-last_area / m) if m < 0 and last_area > 0 else None
        if days_close:
            days_close = max(0, days_close)
            close_date = (datetime.now() + timedelta(days=days_close)).date().isoformat()
        else:
            close_date = None

        pct_change = round((areas[0] - areas[-1]) / max(areas[0], 0.01) * 100, 1)

        traj = []
        for d in range(0, min(90, (days_close or 60) + 10), 7):
            val = max(0.0, float(m * (last_day + d) + b))
            traj.append({"day": d, "area_pct": round(val, 2)})

        return {
            "healing_rate_per_day":      round(float(m), 4),
            "projected_closure_date":    close_date,
            "days_to_closure":           days_close,
            "pct_improvement":           pct_change,
            "trend":                     "improving" if m < 0 else "deteriorating",
            "trajectory":                traj,
            "scans_used":                len(scans),
        }
    except Exception as e:
        log.error(f"Healing velocity error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 4. FRACTAL DIMENSION  (boundary complexity)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fractal_dimension(mask: np.ndarray) -> float:
    """
    Box-counting fractal dimension of the wound boundary.
    Higher = more complex/irregular edge (chronic wounds tend to be higher).
    Range: 1.0 (straight line) – 2.0 (plane-filling).
    """
    try:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 1.0
        cnt = max(contours, key=cv2.contourArea)
        # Draw contour on binary image
        bnd = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(bnd, [cnt], -1, 255, 1)

        # Box counting
        sizes, counts = [], []
        for k in range(1, 7):
            box = 2 ** k
            # Count non-empty boxes
            h, w = bnd.shape
            n = 0
            for y in range(0, h, box):
                for x in range(0, w, box):
                    if bnd[y:y+box, x:x+box].any():
                        n += 1
            if n > 0:
                sizes.append(box)
                counts.append(n)

        if len(sizes) < 2:
            return 1.0

        log_s = np.log(1.0 / np.array(sizes, dtype=float))
        log_c = np.log(np.array(counts, dtype=float))
        slope = np.polyfit(log_s, log_c, 1)[0]
        return round(float(slope), 3)
    except Exception as e:
        log.error(f"Fractal dimension error: {e}")
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. QR CODE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_qr_code(url: str) -> Optional[str]:
    """
    Generate a QR code PNG (base64) linking to the analysis URL.
    Returns None if qrcode package not installed.
    """
    try:
        import qrcode
        from qrcode.image.pure import PyPNGImage
        qr  = qrcode.QRCode(version=2, box_size=6, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf)
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        log.warning("qrcode not installed — pip install qrcode[pil]")
        return None
    except Exception as e:
        log.error(f"QR code error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. EMBEDDING STORE + SIMILARITY SEARCH
# ─────────────────────────────────────────────────────────────────────────────

_EMB_DB = os.getenv("EMB_DB", "/tmp/embeddings.db")

def _init_emb_db():
    conn = sqlite3.connect(_EMB_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id          TEXT PRIMARY KEY,
            patient_id  TEXT,
            scan_date   TEXT,
            wound_type  TEXT,
            area_pct    REAL,
            severity    TEXT,
            embedding   BLOB,
            created_at  TEXT
        )""")
    conn.commit(); conn.close()

_init_emb_db()


def save_embedding(
    scan_id: str,
    embedding: np.ndarray,
    patient_id: str = "",
    scan_date: str = "",
    wound_type: str = "",
    area_pct: float = 0.0,
    severity: str = "",
):
    """Store a SimCLR embedding for future similarity search."""
    try:
        conn = sqlite3.connect(_EMB_DB)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?,?,?)",
            [scan_id, patient_id, scan_date, wound_type, area_pct, severity,
             embedding.astype(np.float32).tobytes(),
             datetime.now(timezone.utc).isoformat()])
        conn.commit(); conn.close()
    except Exception as e:
        log.error(f"Save embedding error: {e}")


def find_similar_wounds(
    query_embedding: np.ndarray,
    top_k: int = 3,
    exclude_id: str = "",
) -> list:
    """
    Find top_k most similar past wounds using cosine similarity.
    Returns list of dicts with similarity score and metadata.
    """
    try:
        conn = sqlite3.connect(_EMB_DB)
        rows = conn.execute(
            "SELECT id, patient_id, scan_date, wound_type, area_pct, severity, embedding "
            "FROM embeddings WHERE id != ?", [exclude_id]).fetchall()
        conn.close()

        if not rows:
            return []

        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        results = []
        for row in rows:
            emb = np.frombuffer(row[6], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(q, emb))
            results.append({
                "scan_id":    row[0],
                "patient_id": row[1],
                "scan_date":  row[2],
                "wound_type": row[3],
                "area_pct":   row[4],
                "severity":   row[5],
                "similarity": round(sim, 4),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    except Exception as e:
        log.error(f"Similarity search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 7. PDF REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    result: dict,
    original_img_b64: str,
    output_path: str = "/tmp/wound_report.pdf",
) -> Optional[str]:
    """
    Generate a clinical PDF report.
    Returns base64-encoded PDF string, or None on failure.

    Requires: pip install reportlab
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage)

        buf   = io.BytesIO()
        doc   = SimpleDocTemplate(buf, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle("title", parent=styles["Title"],
                                      fontSize=18, textColor=colors.HexColor("#0f172a"))
        h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
                                      fontSize=13, textColor=colors.HexColor("#1e40af"),
                                      spaceBefore=12)
        body_style  = styles["BodyText"]

        # Header
        story.append(Paragraph("WoundAI Clinical Report", title_style))
        story.append(Paragraph(
            f"Report ID: {result.get('doctor_report', {}).get('report_id', '—')} &nbsp;|&nbsp; "
            f"Generated: {result.get('timestamp', datetime.now().isoformat())[:19]}",
            body_style))
        story.append(Spacer(1, 0.4*cm))

        # Disclaimer
        disc = ParagraphStyle("disc", parent=body_style, fontSize=8,
                               textColor=colors.red)
        story.append(Paragraph(
            "⚠ RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS. "
            "All findings require clinical verification.", disc))
        story.append(Spacer(1, 0.5*cm))

        # Original image
        if original_img_b64:
            try:
                img_data = base64.b64decode(original_img_b64.split(",")[-1])
                img_buf  = io.BytesIO(img_data)
                rl_img   = RLImage(img_buf, width=6*cm, height=6*cm)
                story.append(rl_img)
                story.append(Spacer(1, 0.3*cm))
            except Exception:
                pass

        # Wound metrics table
        story.append(Paragraph("Wound Metrics", h2_style))
        m = result.get("metrics", {})
        wt = result.get("wound_type", {})
        metrics_data = [
            ["Metric", "Value"],
            ["Wound Type",      wt.get("label", "—")],
            ["ICD-10 Code",     wt.get("icd10", "—")],
            ["Area",            f"{m.get('area_percentage', 0):.2f}% of image"],
            ["Area (mm²)",      str(m.get("area_mm2", "—"))],
            ["Perimeter",       f"{m.get('perimeter', 0):.1f} px"],
            ["Severity",        m.get("severity", "—")],
            ["Circularity",     str(m.get("circularity", "—"))],
            ["Confidence",      f"{m.get('mean_confidence', 0)*100:.1f}%"],
        ]
        push = result.get("push_score", {})
        if push:
            metrics_data.append(["PUSH Score", f"{push.get('total','—')} / 17"])
            metrics_data.append(["PUSH Interpretation", push.get("interpretation","—")])

        t = Table(metrics_data, colWidths=[6*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING",     (0,0), (-1,-1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

        # Skin analysis
        skin = result.get("skin_analysis", {})
        if skin:
            story.append(Paragraph("Skin Analysis", h2_style))
            story.append(Paragraph(
                f"Type: {skin.get('skin_type','—')} &nbsp;|&nbsp; "
                f"ITA: {skin.get('ita_angle',0):.1f}° &nbsp;|&nbsp; "
                f"Method: {skin.get('method','ITA')}", body_style))
            story.append(Spacer(1, 0.3*cm))

        # Healing stage
        h = result.get("healing_stage", {})
        if h:
            story.append(Paragraph("Healing Assessment", h2_style))
            story.append(Paragraph(
                f"<b>Stage:</b> {h.get('stage','—')} "
                f"(confidence {h.get('confidence',0)*100:.0f}%)", body_style))
            story.append(Paragraph(h.get("description",""), body_style))
            story.append(Spacer(1, 0.2*cm))

        # Recommendations
        recs = wt.get("recommendations", []) or h.get("recommendations", [])
        if recs:
            story.append(Paragraph("Clinical Recommendations", h2_style))
            for r in recs:
                story.append(Paragraph(f"• {r}", body_style))
            story.append(Spacer(1, 0.3*cm))

        # Footer
        footer = ParagraphStyle("footer", parent=body_style,
                                 fontSize=7, textColor=colors.grey)
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(
            "WoundAI v4.0 | SimCLR + U-Net | Google Cloud Run | "
            "Research use only — not for clinical diagnosis", footer))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        return base64.b64encode(pdf_bytes).decode()

    except ImportError:
        log.warning("reportlab not installed — pip install reportlab")
        return None
    except Exception as e:
        log.error(f"PDF generation error: {e}")
        return None
