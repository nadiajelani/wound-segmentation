"""
Improvement 3 – Image Quality Gate
====================================
Scores an image on 5 axes and returns a pass/fail decision with
per-axis reasons.  Call this BEFORE predict_with_uncertainty() in app.py.

Integration in app.py
-----------------------
  from image_quality import check_quality, QualityError

  @app.route("/analyze", methods=["POST"])
  def analyze_wound():
      ...
      img_arr = preprocess_image(raw)       # (1,H,W,3) float32
      try:
          check_quality(img_arr[0])         # raises QualityError if bad
      except QualityError as e:
          return jsonify({"error": str(e), "quality_report": e.report}), 422
      ...

Axes checked
------------
  1. Blur         – Laplacian variance  < threshold → too blurry
  2. Brightness   – mean luminance      < 30        → too dark
                                        > 230       → overexposed
  3. Contrast     – std-dev of L*       < 15        → low contrast
  4. Minimum size – longest image side  < 200 px    → too small
  5. Colour cast  – dominant channel    > 90 %      → strong cast
                    (catches pure-red/pure-blue error images)
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger("image_quality")

# ── tuneable thresholds (override via env or config) ──────────────────────────
THRESHOLDS = {
    "blur_min":         80.0,   # Laplacian variance; lower = blurrier
    "brightness_min":   30.0,   # mean pixel value 0-255
    "brightness_max":  230.0,
    "contrast_min":     15.0,   # std-dev in L* channel
    "min_side_px":     200,     # shortest acceptable image dimension
    "colour_cast_max":   0.90,  # max fraction one channel may dominate
}


@dataclass
class QualityReport:
    passed: bool
    score: float                    # 0.0 – 1.0 composite
    issues: List[str] = field(default_factory=list)
    axes: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed":  self.passed,
            "score":   round(self.score, 3),
            "issues":  self.issues,
            "axes":    self.axes,
        }


class QualityError(ValueError):
    """Raised when an image fails quality checks."""
    def __init__(self, message: str, report: QualityReport):
        super().__init__(message)
        self.report = report.to_dict()


def _blur_score(img_u8_gray: np.ndarray) -> float:
    """Higher = sharper.  Laplacian variance method."""
    return float(cv2.Laplacian(img_u8_gray, cv2.CV_64F).var())


def _brightness(img_u8_gray: np.ndarray) -> float:
    return float(np.mean(img_u8_gray))


def _contrast(img_u8_rgb: np.ndarray) -> float:
    """Std-dev of L* channel in CIE L*a*b* space."""
    lab = cv2.cvtColor(img_u8_rgb, cv2.COLOR_RGB2Lab)
    return float(np.std(lab[:, :, 0]))


def _colour_cast(img_u8_rgb: np.ndarray) -> float:
    """Returns the maximum single-channel fraction of total energy."""
    means = img_u8_rgb.mean(axis=(0, 1))
    total = means.sum() + 1e-8
    return float(means.max() / total)


def check_quality(
    img_01: np.ndarray,
    thresholds: dict = None,
    raise_on_fail: bool = True,
) -> QualityReport:
    """
    Parameters
    ----------
    img_01       : float32 array (H, W, 3) normalised to [0,1]
    thresholds   : override any threshold key from THRESHOLDS dict
    raise_on_fail: if True, raises QualityError on failure

    Returns
    -------
    QualityReport (always returned; error only raised if raise_on_fail)
    """
    t = {**THRESHOLDS, **(thresholds or {})}

    img_u8 = (np.clip(img_01, 0, 1) * 255).astype(np.uint8)
    gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    h, w   = img_u8.shape[:2]

    axes   = {}
    issues = []
    scores = []

    # 1. Minimum size
    min_side = min(h, w)
    size_ok  = min_side >= t["min_side_px"]
    axes["size"] = {"value": min_side, "threshold": t["min_side_px"],
                    "passed": size_ok}
    if not size_ok:
        issues.append(f"Image too small ({min_side}px min side, need ≥{t['min_side_px']}px)."
                      " Re-capture closer to the wound.")
    scores.append(min(1.0, min_side / t["min_side_px"]))

    # 2. Blur
    blur_val = _blur_score(gray)
    blur_ok  = blur_val >= t["blur_min"]
    axes["blur"] = {"value": round(blur_val, 1), "threshold": t["blur_min"],
                    "passed": blur_ok}
    if not blur_ok:
        issues.append(f"Image too blurry (sharpness={blur_val:.0f}, need ≥{t['blur_min']})."
                      " Hold the camera still and ensure good focus.")
    scores.append(min(1.0, blur_val / (t["blur_min"] * 2)))

    # 3. Brightness
    bright_val = _brightness(gray)
    bright_ok  = t["brightness_min"] <= bright_val <= t["brightness_max"]
    axes["brightness"] = {"value": round(bright_val, 1),
                           "min": t["brightness_min"],
                           "max": t["brightness_max"],
                           "passed": bright_ok}
    if bright_val < t["brightness_min"]:
        issues.append(f"Image too dark (mean={bright_val:.0f}). Use better lighting.")
    elif bright_val > t["brightness_max"]:
        issues.append(f"Image overexposed (mean={bright_val:.0f}). Reduce direct light.")
    # normalise brightness score: 1 at midpoint, 0 at extremes
    mid   = (t["brightness_min"] + t["brightness_max"]) / 2
    half  = (t["brightness_max"] - t["brightness_min"]) / 2
    scores.append(max(0.0, 1 - abs(bright_val - mid) / half))

    # 4. Contrast
    contrast_val = _contrast(img_u8)
    contrast_ok  = contrast_val >= t["contrast_min"]
    axes["contrast"] = {"value": round(contrast_val, 1),
                         "threshold": t["contrast_min"],
                         "passed": contrast_ok}
    if not contrast_ok:
        issues.append(f"Low image contrast (std={contrast_val:.1f}). "
                      "Ensure wound is well-lit and not covered.")
    scores.append(min(1.0, contrast_val / (t["contrast_min"] * 2)))

    # 5. Colour cast
    cast_val = _colour_cast(img_u8)
    cast_ok  = cast_val <= t["colour_cast_max"]
    axes["colour_cast"] = {"value": round(cast_val, 3),
                            "threshold": t["colour_cast_max"],
                            "passed": cast_ok}
    if not cast_ok:
        issues.append(f"Strong colour cast detected (dominant channel={cast_val:.0%}). "
                      "Check camera white balance or remove coloured light sources.")
    scores.append(max(0.0, 1 - (cast_val - 0.4) / 0.5))

    composite = round(float(np.mean(scores)), 3)
    passed     = len(issues) == 0

    report = QualityReport(passed=passed, score=composite,
                           issues=issues, axes=axes)

    log.info("Quality check: passed=%s score=%.2f issues=%s",
             passed, composite, issues)

    if not passed and raise_on_fail:
        summary = "; ".join(issues)
        raise QualityError(f"Image quality insufficient: {summary}", report)

    return report


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python image_quality.py <image_path>")
        sys.exit(1)

    img_bgr = cv2.imread(sys.argv[1])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_01  = img_rgb.astype(np.float32) / 255.0

    report = check_quality(img_01, raise_on_fail=False)
    print(json.dumps(report.to_dict(), indent=2))
    print("\n✅ PASS" if report.passed else "\n❌ FAIL")
