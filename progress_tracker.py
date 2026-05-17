"""
Improvement 4 – Wound Progress Tracking (Backend)
===================================================
New Flask endpoint  POST /compare
Accepts two images + optional dates, runs analysis on both,
computes delta metrics and a healing velocity forecast.

Add to app.py
--------------
  from progress_tracker import compare_wounds
  app.register_blueprint(compare_wounds)

Returned JSON structure
------------------------
{
  "success": true,
  "scan_a": { ...full analyze result... },
  "scan_b": { ...full analyze result... },
  "delta": {
    "area_px":    -320,
    "area_pct":   -2.1,
    "perimeter":  -18.4,
    "circularity": +0.05,
    "days_elapsed": 7,
    "healing_rate_pct_per_day": -0.30,
    "projected_closure_days": 24,
    "trend": "improving" | "stable" | "deteriorating"
  }
}
"""

import math
import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

log = logging.getLogger("progress_tracker")
compare_wounds = Blueprint("compare_wounds", __name__)


def _parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return datetime.now(timezone.utc)


def compute_delta(metrics_a: dict, metrics_b: dict,
                  date_a: str = None, date_b: str = None) -> dict:
    """
    Compute delta between two wound metric dicts.
    scan_a = older / baseline;  scan_b = more recent.
    """
    da = _parse_date(date_a) if date_a else datetime.now(timezone.utc)
    db = _parse_date(date_b) if date_b else datetime.now(timezone.utc)
    days = max(1, abs((db - da).days))

    area_a    = metrics_a.get("area_percentage",   0.0)
    area_b    = metrics_b.get("area_percentage",   0.0)
    perim_a   = metrics_a.get("perimeter",         0.0)
    perim_b   = metrics_b.get("perimeter",         0.0)
    circ_a    = metrics_a.get("circularity",        0.0)
    circ_b    = metrics_b.get("circularity",        0.0)
    px_a      = metrics_a.get("area_pixels",        0)
    px_b      = metrics_b.get("area_pixels",        0)

    delta_area     = round(area_b - area_a, 3)
    delta_px       = px_b - px_a
    delta_perim    = round(perim_b - perim_a, 2)
    delta_circ     = round(circ_b - circ_a, 3)

    # Healing rate: negative = wound shrinking (good)
    healing_rate   = round(delta_area / days, 4)   # % per day

    # Projected closure (linear extrapolation)
    if healing_rate < 0 and area_b > 0:
        days_to_close = round(area_b / abs(healing_rate))
    else:
        days_to_close = None

    # Trend
    if delta_area < -0.5:    trend = "improving"
    elif delta_area > 0.5:   trend = "deteriorating"
    else:                    trend = "stable"

    # Percentage improvement
    pct_improvement = None
    if area_a > 0:
        pct_improvement = round((area_a - area_b) / area_a * 100, 1)

    return {
        "days_elapsed":              days,
        "area_pct_change":           delta_area,
        "area_px_change":            delta_px,
        "perimeter_change":          delta_perim,
        "circularity_change":        delta_circ,
        "healing_rate_pct_per_day":  healing_rate,
        "projected_closure_days":    days_to_close,
        "pct_improvement":           pct_improvement,
        "trend":                     trend,
        "area_mm2_a":                metrics_a.get("area_mm2"),
        "area_mm2_b":                metrics_b.get("area_mm2"),
    }


@compare_wounds.route("/compare", methods=["POST"])
def compare_endpoint():
    """
    Accepts multipart/form-data:
      image_a   – older wound image (file)
      image_b   – newer wound image (file)
      date_a    – ISO date string (optional)
      date_b    – ISO date string (optional)
    """
    # Import here to avoid circular import when module is used standalone
    try:
        from app import preprocess_image, predict_with_uncertainty  # noqa
        from app import calculate_metrics, detect_skin_tone          # noqa
        from app import classify_healing_stage, classify_tissue_types # noqa
        from app import generate_doctor_report, make_heatmap         # noqa
        from app import make_overlay, make_contour_overlay, _to_b64_png # noqa
    except ImportError as e:
        return jsonify({"error": f"Could not import app modules: {e}"}), 500

    if "image_a" not in request.files or "image_b" not in request.files:
        return jsonify({"error": "Both image_a and image_b are required"}), 400

    date_a = request.form.get("date_a")
    date_b = request.form.get("date_b")

    results = {}
    for key in ("a", "b"):
        raw  = request.files[f"image_{key}"].read()
        arr  = preprocess_image(raw)
        pm, std, mask = predict_with_uncertainty(arr)
        metrics  = calculate_metrics(mask, pm, std)
        skin     = detect_skin_tone(arr[0])
        tissue   = classify_tissue_types(arr[0], mask)
        healing  = classify_healing_stage(metrics, tissue)
        ts       = date_a if key == "a" else date_b or datetime.now(timezone.utc).isoformat()
        report   = generate_doctor_report(metrics, healing, skin, tissue, ts)
        heatmap  = make_heatmap(pm)
        overlay  = make_overlay(arr[0], heatmap)
        contour  = make_contour_overlay(arr[0], mask)

        results[key] = {
            "metrics":           metrics,
            "skin_analysis":     skin,
            "tissue_composition":tissue,
            "healing_stage":     healing,
            "doctor_report":     report,
            "mask_image":        _to_b64_png(mask),
            "heatmap_image":     _to_b64_png(heatmap),
            "overlay_image":     _to_b64_png(overlay),
            "contour_image":     _to_b64_png(contour),
        }

    delta = compute_delta(
        results["a"]["metrics"], results["b"]["metrics"],
        date_a, date_b
    )

    return jsonify({
        "success": True,
        "scan_a":  results["a"],
        "scan_b":  results["b"],
        "delta":   delta,
    })
