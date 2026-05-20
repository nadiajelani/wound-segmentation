"""
apply_features.py
------------------
Run once from your project root to wire all new features into app.py:

    python apply_features.py

Adds to every /analyze response:
  gradcam_image     – Grad-CAM saliency map (base64 PNG)
  gradcam_overlay   – Grad-CAM blended over original image
  embedding         – 2048-d SimCLR encoder vector (for similarity)
  edge_sharpness    – boundary confidence score
  convexity_defects – undermining risk
  satellite_lesions – multi-region analysis
  healing_score     – composite 0-100 healing probability
  texture           – GLCM texture features
  orientation       – ellipse fit, aspect ratio, wound type hint
"""

import re, shutil
from pathlib import Path

APP = Path("app.py")
if not APP.exists():
    print("❌ app.py not found — run from project root")
    exit(1)

shutil.copy(APP, "app_before_features.py")
print("✅ Backed up to app_before_features.py")

src = APP.read_text()

# ── PATCH 1: Add import at top ────────────────────────────────────────────────
IMPORT_MARKER = "import numpy as np"
IMPORT_NEW = """import numpy as np
try:
    from gradcam import (
        make_gradcam, make_gradcam_overlay, extract_embedding,
        wound_similarity, edge_sharpness, convexity_defect_score,
        satellite_lesions, healing_score, texture_features, wound_orientation,
    )
    GRADCAM_ENABLED = True
except ImportError as _e:
    GRADCAM_ENABLED = False
    import logging as _log
    _log.getLogger("woundai").warning(f"gradcam.py not found — advanced features disabled: {_e}")"""

if IMPORT_MARKER in src and "from gradcam import" not in src:
    src = src.replace(IMPORT_MARKER, IMPORT_NEW, 1)
    print("✅ Patch 1: gradcam imports added")
else:
    print("⚠️  Patch 1: already applied or import marker not found")

# ── PATCH 2: Wire features into analyze_wound ─────────────────────────────────
# Find the block that builds visuals and return JSON — insert after it
OLD_RESULT_BUILD = """        # Build visuals
        logger.info(f"Building heatmap from pred_map shape: {pred_map.shape}, dtype: {pred_map.dtype}")
        heatmap_bgr = make_heatmap(pred_map)                          # HxWx3 (BGR)
        logger.info(f"Heatmap created: shape {heatmap_bgr.shape}, dtype {heatmap_bgr.dtype}")
        
        logger.info(f"Building overlay from image shape: {img_array[0].shape}")
        overlay_bgr = make_overlay(img_array[0], heatmap_bgr, 0.45)   # HxWx3 (BGR)
        logger.info(f"Overlay created: shape {overlay_bgr.shape}, dtype {overlay_bgr.dtype}")
        
        # Encode images
        logger.info("Encoding images to base64...")
        mask_b64     = to_base64_png(mask)          # grayscale
        heatmap_b64  = to_base64_png(heatmap_bgr)   # color heatmap
        overlay_b64  = to_base64_png(overlay_bgr)   # blended on original
        
        logger.info(f"Image encoding complete:")
        logger.info(f"  - mask_b64 length: {len(mask_b64)}")
        logger.info(f"  - heatmap_b64 length: {len(heatmap_b64)}")
        logger.info(f"  - overlay_b64 length: {len(overlay_b64)}")
        
        result = {
            "success": True,
            "metrics": metrics,
            "skin_analysis": skin_analysis,  # NEW: Skin tone information
            "healing_stage": healing_stage,
            "doctor_report": doctor_report,
            "mask_image": mask_b64,
            "heatmap_image": heatmap_b64,
            "overlay_image": overlay_b64,
            "timestamp": timestamp
        }"""

NEW_RESULT_BUILD = """        # Build visuals
        logger.info(f"Building heatmap from pred_map shape: {pred_map.shape}, dtype: {pred_map.dtype}")
        heatmap_bgr = make_heatmap(pred_map)
        logger.info(f"Heatmap created: shape {heatmap_bgr.shape}, dtype {heatmap_bgr.dtype}")

        logger.info(f"Building overlay from image shape: {img_array[0].shape}")
        overlay_bgr = make_overlay(img_array[0], heatmap_bgr, 0.45)
        logger.info(f"Overlay created: shape {overlay_bgr.shape}, dtype {overlay_bgr.dtype}")

        # Contour overlay
        try:
            import cv2 as _cv
            contours_img = (img_array[0] * 255).astype('uint8')[:, :, ::-1].copy()
            cnts, _ = _cv.findContours(mask, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
            _cv.drawContours(contours_img, cnts, -1, (0, 220, 80), 2)
            contour_b64 = to_base64_png(contours_img)
        except Exception:
            contour_b64 = None

        # ── Advanced features from gradcam.py ──────────────────────────────
        gradcam_b64 = gradcam_overlay_b64 = None
        embedding   = None
        edge_info   = {}
        defect_info = {}
        satellite   = {}
        heal_score  = {}
        texture     = {}
        orientation = {}

        if GRADCAM_ENABLED:
            try:
                cam_bgr, gradcam_b64 = make_gradcam(MODEL, img_array)
                _, gradcam_overlay_b64 = make_gradcam_overlay(img_array, cam_bgr)
                logger.info("Grad-CAM computed ✅")
            except Exception as _e:
                logger.warning(f"Grad-CAM failed: {_e}")

            try:
                embedding = extract_embedding(MODEL, img_array)
                logger.info("Embedding extracted ✅")
            except Exception as _e:
                logger.warning(f"Embedding failed: {_e}")

            try:
                edge_info   = edge_sharpness(pred_map, mask)
                defect_info = convexity_defect_score(mask)
                satellite   = satellite_lesions(mask)
                heal_score  = healing_score(metrics, edge_info, defect_info)
                texture     = texture_features(img_array[0], mask)
                orientation = wound_orientation(mask)
                logger.info("Shape analytics computed ✅")
            except Exception as _e:
                logger.warning(f"Shape analytics failed: {_e}")

        # Add advanced fields to metrics
        metrics["edge_sharpness"]      = edge_info.get("mean_sharpness", None)
        metrics["boundary_confidence"] = edge_info.get("boundary_confidence", None)
        metrics["undermining_risk"]    = defect_info.get("undermining_risk", None)
        metrics["defect_count"]        = defect_info.get("defect_count", None)
        metrics["aspect_ratio"]        = orientation.get("aspect_ratio", None)
        metrics["orientation_deg"]     = orientation.get("orientation_deg", None)
        metrics["wound_type_hint"]     = orientation.get("wound_type_hint", None)

        # Encode standard images
        logger.info("Encoding images to base64...")
        mask_b64    = to_base64_png(mask)
        heatmap_b64 = to_base64_png(heatmap_bgr)
        overlay_b64 = to_base64_png(overlay_bgr)

        logger.info(f"Image encoding complete — mask:{len(mask_b64)} heatmap:{len(heatmap_b64)}")

        result = {
            "success":            True,
            "metrics":            metrics,
            "skin_analysis":      skin_analysis,
            "healing_stage":      healing_stage,
            "doctor_report":      doctor_report,
            "satellite_lesions":  satellite,
            "healing_score":      heal_score,
            "texture_features":   texture,
            "mask_image":         mask_b64,
            "heatmap_image":      heatmap_b64,
            "overlay_image":      overlay_b64,
            "contour_image":      contour_b64,
            "gradcam_image":      gradcam_b64,
            "gradcam_overlay":    gradcam_overlay_b64,
            "embedding":          embedding.tolist() if embedding is not None else None,
            "timestamp":          timestamp,
            "version":            "4.0.0",
        }"""

if OLD_RESULT_BUILD in src:
    src = src.replace(OLD_RESULT_BUILD, NEW_RESULT_BUILD)
    print("✅ Patch 2: analyze_wound wired with all new features")
else:
    print("⚠️  Patch 2: could not find exact result-build block — manual integration needed")
    print("    → Copy NEW_RESULT_BUILD from this script into app.py manually")
    print("    → Replace the block starting with '# Build visuals'")

# ── PATCH 3: IMG_SIZE fix (224 not 128) ──────────────────────────────────────
for old_sz, new_sz in [
    ("target_size=(128, 128)", "target_size=(224, 224)"),
    ("image.resize((128, 128)", "image.resize((224, 224)"),
    ('os.getenv("IMG_H", "128")', 'os.getenv("IMG_H", "224")'),
    ('os.getenv("IMG_W", "128")', 'os.getenv("IMG_W", "224")'),
]:
    if old_sz in src:
        src = src.replace(old_sz, new_sz)
        print(f"✅ Patch 3: {old_sz} → {new_sz}")

# ── PATCH 4: Fix threshold (adaptive, not fixed 0.5) ─────────────────────────
OLD_THRESH = "mask = (pred_map > 0.5).astype(np.uint8) * 255"
NEW_THRESH = """# Adaptive Otsu threshold — fixes circular mask on clean wounds
        import cv2 as _cv2
        _u8 = (pred_map * 255).astype(np.uint8)
        _otsu, _ = _cv2.threshold(_u8, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
        _thresh = max(0.20, min(0.65, float(_otsu) / 255.0))
        _raw_mask = (pred_map >= _thresh).astype(np.uint8) * 255
        # Morphological cleanup
        _k_o = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3,3))
        _k_c = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5,5))
        _raw_mask = _cv2.morphologyEx(_raw_mask, _cv2.MORPH_OPEN,  _k_o, iterations=1)
        mask = _cv2.morphologyEx(_raw_mask,  _cv2.MORPH_CLOSE, _k_c, iterations=2)"""
if OLD_THRESH in src:
    src = src.replace(OLD_THRESH, NEW_THRESH)
    print("✅ Patch 4: Adaptive Otsu threshold applied")

# ── PATCH 5: Fix health check version string ─────────────────────────────────
src = src.replace('"version": "2.0.0-HEATMAP-ENABLED"', '"version": "4.0.0"')
src = src.replace('"version": "3.0.0"', '"version": "4.0.0"')
print("✅ Patch 5: Version updated to 4.0.0")

APP.write_text(src)
print("\n✅ app.py patched — all features enabled")
print("\nNext steps:")
print("  1. Make sure gradcam.py is in the same folder as app.py")
print("  2. pip install scikit-image  (for texture features)")
print("  3. python app.py  to test locally")
print("  4. gcloud run deploy wound-api --source . --region us-central1 --memory 4Gi --cpu 2 --timeout 600 --allow-unauthenticated")
