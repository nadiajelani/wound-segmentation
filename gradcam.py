"""
gradcam.py  –  SimCLR U-Net Feature Extraction Module
======================================================
Drop this file next to app.py.  Import in app.py with:

    from gradcam import (
        make_gradcam,
        extract_embedding,
        wound_similarity,
        edge_sharpness,
        convexity_defect_score,
        satellite_lesions,
        healing_score,
        texture_features,
    )

All functions are stateless and accept the same inputs your existing
app.py already produces — no new preprocessing needed.

Inputs (consistent throughout):
    model       – the loaded Keras model  (global MODEL from app.py)
    img_arr     – float32 (1, H, W, 3) from preprocess_image()
    pred_map    – float32 (H, W)        from predict_wound_mask()
    mask        – uint8  (H, W) {0,255} from predict_wound_mask()
"""

import math
import logging
import numpy as np
import cv2
import io
import base64
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger("woundai.gradcam")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_b64_png(arr: np.ndarray) -> str:
    """Encode uint8 numpy array to base64 PNG string (no data: prefix)."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────

def make_gradcam(
    model,
    img_arr: np.ndarray,
    layer_name: str = "conv5_block3_out",
) -> Tuple[np.ndarray, str]:
    """
    Produces a Grad-CAM saliency map showing which image regions
    the SimCLR encoder focused on when making its segmentation decision.

    Parameters
    ----------
    model       : loaded Keras model
    img_arr     : float32 (1, H, W, 3)
    layer_name  : ResNet50 layer to hook — conv5_block3_out is the
                  deepest feature map before the U-Net decoder

    Returns
    -------
    cam_bgr     : uint8 (H, W, 3) BGR heatmap — use for display
    cam_b64     : base64 PNG string
    """
    try:
        import tensorflow as tf

        H, W = img_arr.shape[1], img_arr.shape[2]

        # Find the target layer
        target_layer = None
        for lyr in model.layers:
            if lyr.name == layer_name:
                target_layer = lyr
                break

        # Fallback: use the last Conv2D layer before the bottleneck
        if target_layer is None:
            for lyr in reversed(model.layers):
                if isinstance(lyr, tf.keras.layers.Conv2D):
                    target_layer = lyr
                    logger.warning(f"Layer '{layer_name}' not found, using '{lyr.name}'")
                    break

        if target_layer is None:
            raise ValueError("No suitable convolutional layer found for Grad-CAM")

        # Build a sub-model: input → [target_layer_output, final_output]
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        tensor = tf.cast(img_arr, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(tensor)
            conv_outputs, predictions = grad_model(tensor, training=False)
            # For segmentation: use mean of the output probability map as the scalar
            loss = tf.reduce_mean(predictions)

        # Gradients of loss w.r.t. conv feature map
        grads = tape.gradient(loss, conv_outputs)           # (1, fH, fW, C)

        # Pool gradients over spatial dimensions → weight per channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        # Weight feature maps by pooled gradients
        conv_out = conv_outputs[0]                          # (fH, fW, C)
        cam = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_out), axis=-1
        ).numpy()                                           # (fH, fW)

        # ReLU + normalise
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        cam = (cam * 255).astype(np.uint8)

        # Resize to original image size
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

        # Apply INFERNO colormap (more readable than JET for explainability)
        cam_bgr = cv2.applyColorMap(cam, cv2.COLORMAP_INFERNO)

        logger.info(f"Grad-CAM computed from layer '{target_layer.name}'")
        return cam_bgr, _to_b64_png(cam_bgr)

    except Exception as exc:
        logger.error(f"Grad-CAM failed: {exc}")
        # Return blank image so the rest of the pipeline doesn't break
        blank = np.zeros((img_arr.shape[1], img_arr.shape[2], 3), dtype=np.uint8)
        return blank, _to_b64_png(blank)


def make_gradcam_overlay(
    img_arr: np.ndarray,
    cam_bgr: np.ndarray,
    alpha: float = 0.55,
) -> Tuple[np.ndarray, str]:
    """
    Blend Grad-CAM over the original image.

    Returns
    -------
    overlay_bgr : uint8 (H, W, 3)
    overlay_b64 : base64 PNG
    """
    base_bgr = (img_arr[0] * 255).astype(np.uint8)[:, :, ::-1]
    cam_r    = _resize_to(cam_bgr, base_bgr.shape[0], base_bgr.shape[1])
    blended  = cv2.addWeighted(cam_r, alpha, base_bgr, 1 - alpha, 0)
    return blended, _to_b64_png(blended)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENCODER EMBEDDING  (for similarity search + drift tracking)
# ─────────────────────────────────────────────────────────────────────────────

_ENCODER_MODEL = None   # cached sub-model


def extract_embedding(model, img_arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the 2048-d SimCLR embedding from conv5_block3_out.

    The embedding is L2-normalised so cosine similarity = dot product.

    Returns
    -------
    embedding : float32 (2048,) or None on failure
    """
    global _ENCODER_MODEL
    try:
        import tensorflow as tf

        if _ENCODER_MODEL is None:
            # Build encoder sub-model once and cache it
            bottleneck = None
            for lyr in model.layers:
                if lyr.name == "conv5_block3_out":
                    bottleneck = lyr
                    break
            if bottleneck is None:
                for lyr in reversed(model.layers):
                    if hasattr(lyr, "output") and len(lyr.output.shape) == 4:
                        bottleneck = lyr
                        break
            if bottleneck is None:
                return None
            _ENCODER_MODEL = tf.keras.Model(
                inputs=model.inputs,
                outputs=bottleneck.output
            )
            logger.info(f"Encoder sub-model built from layer '{bottleneck.name}'")

        feat = _ENCODER_MODEL(tf.cast(img_arr, tf.float32), training=False)
        # Global average pool → (2048,)
        emb = tf.reduce_mean(feat, axis=[1, 2]).numpy()[0]
        # L2 normalise
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    except Exception as exc:
        logger.error(f"Embedding extraction failed: {exc}")
        return None


def wound_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised embeddings.
    1.0 = identical, 0.0 = orthogonal, -1.0 = opposite.

    Use case: "How similar is this wound to the previous scan?"
    """
    if emb_a is None or emb_b is None:
        return 0.0
    return float(np.dot(emb_a, emb_b))


# ─────────────────────────────────────────────────────────────────────────────
# 3. EDGE SHARPNESS SCORE
# ─────────────────────────────────────────────────────────────────────────────

def edge_sharpness(pred_map: np.ndarray, mask: np.ndarray) -> dict:
    """
    Gradient magnitude at the wound boundary.

    High sharpness → model is confident about the exact edge location.
    Low sharpness  → uncertain/blurry boundary (may indicate necrotic edge
                     or imaging artefact).

    Returns
    -------
    {
      "mean_sharpness": float,      # 0–1
      "boundary_confidence": str,   # "high" | "medium" | "low"
    }
    """
    try:
        gy, gx = np.gradient(pred_map)
        magnitude = np.sqrt(gx**2 + gy**2)

        # Sample only boundary pixels (within 3px of contour)
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated  = cv2.dilate(mask, kernel)
        eroded   = cv2.erode(mask, kernel)
        boundary = (dilated > 0) & (eroded == 0)

        if boundary.sum() == 0:
            return {"mean_sharpness": 0.0, "boundary_confidence": "low"}

        boundary_grad = magnitude[boundary]
        mean_sharp    = float(np.mean(boundary_grad))
        norm_sharp    = min(1.0, mean_sharp * 8.0)   # scale to ~0-1

        if norm_sharp > 0.6:
            level = "high"
        elif norm_sharp > 0.3:
            level = "medium"
        else:
            level = "low"

        return {
            "mean_sharpness":      round(norm_sharp, 3),
            "boundary_confidence": level,
        }
    except Exception as exc:
        logger.error(f"Edge sharpness failed: {exc}")
        return {"mean_sharpness": 0.0, "boundary_confidence": "unknown"}


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONVEXITY DEFECTS  (undermined edges)
# ─────────────────────────────────────────────────────────────────────────────

def convexity_defect_score(mask: np.ndarray) -> dict:
    """
    Detects points where the wound boundary dips inward (undermining).

    Clinical significance:
      • High defect count → irregular/undermined wound (pressure ulcer)
      • Low defect count  → clean regular wound (surgical incision)

    Returns
    -------
    {
      "defect_count":   int,
      "max_depth_px":   float,
      "mean_depth_px":  float,
      "undermining_risk": "none" | "low" | "moderate" | "high"
    }
    """
    try:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"defect_count": 0, "max_depth_px": 0.0,
                    "mean_depth_px": 0.0, "undermining_risk": "none"}

        cnt  = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt, returnPoints=False)

        if hull is None or len(hull) < 4:
            return {"defect_count": 0, "max_depth_px": 0.0,
                    "mean_depth_px": 0.0, "undermining_risk": "none"}

        defects = cv2.convexityDefects(cnt, hull)
        if defects is None:
            return {"defect_count": 0, "max_depth_px": 0.0,
                    "mean_depth_px": 0.0, "undermining_risk": "none"}

        # depth is stored as fixed-point ÷ 256
        depths = [d[0][3] / 256.0 for d in defects if d[0][3] / 256.0 > 2.0]

        count      = len(depths)
        max_depth  = float(max(depths)) if depths else 0.0
        mean_depth = float(np.mean(depths)) if depths else 0.0

        if count == 0:          risk = "none"
        elif count <= 3:        risk = "low"
        elif count <= 8:        risk = "moderate"
        else:                   risk = "high"

        return {
            "defect_count":     count,
            "max_depth_px":     round(max_depth, 1),
            "mean_depth_px":    round(mean_depth, 1),
            "undermining_risk": risk,
        }
    except Exception as exc:
        logger.error(f"Convexity defects failed: {exc}")
        return {"defect_count": 0, "max_depth_px": 0.0,
                "mean_depth_px": 0.0, "undermining_risk": "unknown"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. SATELLITE LESION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def satellite_lesions(mask: np.ndarray, min_area_px: int = 20) -> dict:
    """
    Detect multiple separate wound regions in a single image.

    Each region gets its own area, perimeter, and severity.

    Returns
    -------
    {
      "count": int,
      "regions": [
          {
            "id": int,
            "area_pixels": int,
            "area_percentage": float,
            "perimeter": float,
            "severity": str,
            "centroid": [x, y],
          }, ...
      ],
      "primary_region_id": int,   # largest region
    }
    """
    try:
        total_px  = mask.shape[0] * mask.shape[1]
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        regions = []
        for i in range(1, n_labels):   # skip background (label 0)
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue

            # Re-extract contour for this region only
            region_mask = ((labels == i) * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perim = float(cv2.arcLength(cnts[0], True)) if cnts else 0.0
            area_pct = round(area / total_px * 100, 3)

            if area_pct < 1:    sev = "Mild"
            elif area_pct < 5:  sev = "Moderate"
            else:               sev = "Severe"

            regions.append({
                "id":             i,
                "area_pixels":    area,
                "area_percentage": area_pct,
                "perimeter":      round(perim, 1),
                "severity":       sev,
                "centroid":       [round(centroids[i][0], 1),
                                   round(centroids[i][1], 1)],
            })

        regions.sort(key=lambda r: r["area_pixels"], reverse=True)
        primary_id = regions[0]["id"] if regions else None

        return {
            "count":             len(regions),
            "regions":           regions,
            "primary_region_id": primary_id,
            "has_satellites":    len(regions) > 1,
        }
    except Exception as exc:
        logger.error(f"Satellite lesion detection failed: {exc}")
        return {"count": 1, "regions": [], "primary_region_id": None,
                "has_satellites": False}


# ─────────────────────────────────────────────────────────────────────────────
# 6. HEALING PROBABILITY SCORE  (0–100)
# ─────────────────────────────────────────────────────────────────────────────

def healing_score(
    metrics:    dict,
    edge_info:  dict,
    defect_info: dict,
) -> dict:
    """
    Single composite 0–100 score summarising healing likelihood.

    Factors (each 0–1, weighted sum):
      • Model confidence       25 %
      • Wound size (inverse)   25 %
      • Circularity            20 %
      • Edge sharpness         15 %
      • Undermining (inverse)  15 %

    Returns
    -------
    {
      "score":       int,       # 0–100  (higher = more likely to heal well)
      "grade":       str,       # "Excellent" | "Good" | "Fair" | "Poor"
      "factors":     dict,      # breakdown per factor
    }
    """
    try:
        conf  = metrics.get("mean_confidence",   0.75)
        area  = metrics.get("area_percentage",   5.0)
        circ  = metrics.get("circularity",       0.5)
        sharp = edge_info.get("mean_sharpness",  0.5)
        umine = {"none": 0.0, "low": 0.25,
                 "moderate": 0.6, "high": 1.0}.get(
                     defect_info.get("undermining_risk", "none"), 0.0)

        # Normalise area: 0 % → 1.0 (tiny wound), 20 %+ → 0.0 (huge wound)
        area_score = max(0.0, 1.0 - area / 20.0)
        # Undermining penalises healing
        uminv = 1.0 - umine

        weighted = (
            conf       * 0.25 +
            area_score * 0.25 +
            circ       * 0.20 +
            sharp      * 0.15 +
            uminv      * 0.15
        )
        score = int(round(weighted * 100))

        if   score >= 75: grade = "Excellent"
        elif score >= 55: grade = "Good"
        elif score >= 35: grade = "Fair"
        else:             grade = "Poor"

        return {
            "score": score,
            "grade": grade,
            "factors": {
                "model_confidence":  round(conf,       3),
                "wound_size_score":  round(area_score, 3),
                "circularity":       round(circ,       3),
                "edge_sharpness":    round(sharp,      3),
                "undermining_score": round(uminv,      3),
            }
        }
    except Exception as exc:
        logger.error(f"Healing score failed: {exc}")
        return {"score": 50, "grade": "Unknown", "factors": {}}


# ─────────────────────────────────────────────────────────────────────────────
# 7. TEXTURE FEATURES  (GLCM — granularity / homogeneity inside wound)
# ─────────────────────────────────────────────────────────────────────────────

def texture_features(img_rgb_01: np.ndarray, mask: np.ndarray) -> dict:
    """
    Gray-Level Co-occurrence Matrix features inside the wound region.

    Features:
      contrast    – local intensity variation  (high = rough/granular)
      homogeneity – smoothness               (high = uniform/slough)
      energy      – texture regularity       (high = regular pattern)
      correlation – linear dependencies      (high = structured tissue)

    These change as the wound heals even when area stays the same —
    useful as a longitudinal healing marker.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops

        img_u8  = (np.clip(img_rgb_01, 0, 1) * 255).astype(np.uint8)
        gray    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

        wound_pixels = gray[mask > 0]
        if len(wound_pixels) < 50:
            return {}

        # Build a small 2D patch from wound pixels for GLCM
        side = int(math.ceil(math.sqrt(len(wound_pixels))))
        padded = np.zeros(side * side, dtype=np.uint8)
        padded[:len(wound_pixels)] = wound_pixels
        patch = padded.reshape(side, side)

        glcm = graycomatrix(patch, distances=[1], angles=[0],
                            levels=256, symmetric=True, normed=True)

        return {
            "contrast":    round(float(graycoprops(glcm, "contrast")[0, 0]),    3),
            "homogeneity": round(float(graycoprops(glcm, "homogeneity")[0, 0]), 3),
            "energy":      round(float(graycoprops(glcm, "energy")[0, 0]),      3),
            "correlation": round(float(graycoprops(glcm, "correlation")[0, 0]), 3),
        }
    except ImportError:
        logger.warning("scikit-image not installed — texture features skipped")
        return {}
    except Exception as exc:
        logger.error(f"Texture features failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 8. ORIENTATION + ASPECT RATIO
# ─────────────────────────────────────────────────────────────────────────────

def wound_orientation(mask: np.ndarray) -> dict:
    """
    Fit an ellipse to the wound contour and extract:
      • orientation_deg  – angle of major axis (0–180°)
      • aspect_ratio     – major / minor axis length
      • wound_type_hint  – "elongated" | "circular" | "irregular"
    """
    try:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {}
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return {}

        (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
        if mi < 1e-3:
            return {}
        ratio = round(float(ma / mi), 2)

        if ratio > 2.5:   hint = "elongated"
        elif ratio < 1.4: hint = "circular"
        else:             hint = "irregular"

        return {
            "orientation_deg": round(float(angle), 1),
            "aspect_ratio":    ratio,
            "major_axis_px":   round(float(ma), 1),
            "minor_axis_px":   round(float(mi), 1),
            "wound_type_hint": hint,
        }
    except Exception as exc:
        logger.error(f"Wound orientation failed: {exc}")
        return {}
