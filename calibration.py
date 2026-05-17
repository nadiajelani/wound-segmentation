"""
Improvement 2 – Real-World Area Calibration (mm²)
===================================================
Detects a known reference object in the wound image and converts
pixel measurements to physical units (mm², mm).

Supported reference objects
----------------------------
  1. Coin         – Australian 5c (17.65 mm), 10c (23.6 mm), 20c (28.65 mm),
                    50c (31.65 mm), $1 (25 mm), $2 (20.5 mm);
                    US quarter (24.26 mm), dime (17.91 mm)
  2. ArUco marker – 4×4_50 dictionary, known physical size passed as arg
  3. Ruler strip  – horizontal black/white reference strip (fallback)

Drop-in integration with app.py
---------------------------------
  from calibration import calibrate_image, apply_calibration

  px_per_mm, ref_info = calibrate_image(img_rgb, method="auto")
  metrics = apply_calibration(metrics_dict, px_per_mm)
"""

import cv2
import numpy as np
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

log = logging.getLogger("calibration")

# ── known coin diameters in mm ─────────────────────────────────────────────────
COIN_DIAMETERS_MM = {
    "AU_5c":   17.65,
    "AU_10c":  23.60,
    "AU_20c":  28.65,
    "AU_50c":  31.65,
    "AU_$1":   25.00,
    "AU_$2":   20.50,
    "US_quarter": 24.26,
    "US_dime":    17.91,
    "US_nickel":  21.21,
    "US_penny":   19.05,
}

@dataclass
class CalibrationResult:
    px_per_mm: float            # pixels per millimetre
    method: str                 # "aruco" | "coin" | "manual" | "none"
    reference_diameter_mm: Optional[float] = None
    reference_diameter_px: Optional[float] = None
    confidence: float = 1.0     # 0-1
    debug_img: Optional[np.ndarray] = None  # annotated image for UI


# ── ArUco detection ────────────────────────────────────────────────────────────
def detect_aruco(img_bgr: np.ndarray,
                 marker_size_mm: float = 30.0) -> Optional[CalibrationResult]:
    """
    Detect a printed ArUco marker (4×4_50).
    marker_size_mm: the printed physical size of one side of the marker.
    """
    try:
        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(img_bgr)
    except AttributeError:
        # OpenCV < 4.7 fallback
        aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(img_bgr, aruco_dict,
                                                    parameters=aruco_params)

    if ids is None or len(corners) == 0:
        return None

    # Use first detected marker
    c = corners[0][0]  # shape (4, 2)
    side_px = float(np.mean([
        np.linalg.norm(c[0] - c[1]),
        np.linalg.norm(c[1] - c[2]),
        np.linalg.norm(c[2] - c[3]),
        np.linalg.norm(c[3] - c[0]),
    ]))

    px_per_mm = side_px / marker_size_mm

    # Annotate debug image
    debug = img_bgr.copy()
    cv2.polylines(debug, [c.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.putText(debug, f"ArUco {side_px:.0f}px={marker_size_mm}mm",
                tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    log.info("ArUco calibration: %.1f px/mm", px_per_mm)
    return CalibrationResult(
        px_per_mm=px_per_mm,
        method="aruco",
        reference_diameter_mm=marker_size_mm,
        reference_diameter_px=side_px,
        confidence=0.97,
        debug_img=debug,
    )


# ── Coin detection ─────────────────────────────────────────────────────────────
def detect_coin(img_bgr: np.ndarray,
                coin_diameter_mm: float = 25.0) -> Optional[CalibrationResult]:
    """
    Detect a circular coin using Hough circles on a blurred grayscale image.
    coin_diameter_mm: physical diameter of the reference coin.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w    = gray.shape

    # Search for circles in a broad range of radii
    min_r = max(10, min(h, w) // 30)
    max_r = min(h, w) // 4

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 6,
        param1=60,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        log.debug("No coin candidate found")
        return None

    # Pick the most circular candidate (highest accumulator score = first)
    circles = np.round(circles[0]).astype(int)
    best    = circles[0]  # (x, y, r)
    diameter_px = best[2] * 2.0
    px_per_mm   = diameter_px / coin_diameter_mm

    debug = img_bgr.copy()
    cv2.circle(debug, (best[0], best[1]), best[2], (255, 200, 0), 2)
    cv2.putText(debug, f"Coin {diameter_px:.0f}px={coin_diameter_mm}mm",
                (best[0] - 60, best[1] - best[2] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

    log.info("Coin calibration: %.1f px/mm", px_per_mm)
    return CalibrationResult(
        px_per_mm=px_per_mm,
        method="coin",
        reference_diameter_mm=coin_diameter_mm,
        reference_diameter_px=diameter_px,
        confidence=0.85,
        debug_img=debug,
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def calibrate_image(
    img_rgb: np.ndarray,
    method: str = "auto",
    coin_diameter_mm: float = 25.0,
    aruco_marker_size_mm: float = 30.0,
) -> CalibrationResult:
    """
    Try to extract a pixel→mm calibration from the image.

    Parameters
    ----------
    img_rgb           : uint8 RGB image (H×W×3)
    method            : "auto" | "aruco" | "coin" | "none"
    coin_diameter_mm  : diameter of the reference coin if known
    aruco_marker_size_mm : physical side length of the ArUco marker

    Returns
    -------
    CalibrationResult with px_per_mm=None if detection failed
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if method in ("auto", "aruco"):
        result = detect_aruco(img_bgr, aruco_marker_size_mm)
        if result:
            return result

    if method in ("auto", "coin"):
        result = detect_coin(img_bgr, coin_diameter_mm)
        if result:
            return result

    log.warning("Calibration failed – returning uncalibrated result")
    return CalibrationResult(px_per_mm=0.0, method="none", confidence=0.0)


def apply_calibration(metrics: dict, cal: CalibrationResult) -> dict:
    """
    Extend a metrics dict with real-world measurements.
    Adds: area_mm2, perimeter_mm, px_per_mm, calibration_method
    """
    out = dict(metrics)
    if cal.px_per_mm and cal.px_per_mm > 0:
        area_px   = metrics.get("area_pixels", 0)
        perim_px  = metrics.get("perimeter",   0)
        ppm2      = cal.px_per_mm ** 2
        out["area_mm2"]            = round(area_px / ppm2, 2)
        out["perimeter_mm"]        = round(perim_px / cal.px_per_mm, 2)
        out["px_per_mm"]           = round(cal.px_per_mm, 3)
        out["calibration_method"]  = cal.method
        out["calibration_confidence"] = cal.confidence
    else:
        out["area_mm2"]           = None
        out["perimeter_mm"]       = None
        out["px_per_mm"]          = None
        out["calibration_method"] = "none"
        out["calibration_confidence"] = 0.0
    return out


# ── CLI demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python calibration.py <image_path> [coin_mm]")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    coin_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cal = calibrate_image(img_rgb, coin_diameter_mm=coin_mm)

    print(f"Method:     {cal.method}")
    print(f"px/mm:      {cal.px_per_mm:.3f}")
    print(f"Confidence: {cal.confidence:.2f}")

    dummy_metrics = {"area_pixels": 5000, "perimeter": 350}
    result = apply_calibration(dummy_metrics, cal)
    print(f"Area:       {result['area_mm2']} mm²")
    print(f"Perimeter:  {result['perimeter_mm']} mm")

    if cal.debug_img is not None:
        cv2.imwrite("calibration_debug.png", cal.debug_img)
        print("Debug image saved: calibration_debug.png")
