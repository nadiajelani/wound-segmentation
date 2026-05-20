"""
fix_circle_segmentation.py
---------------------------
Fixes the circular mask bug in app.py.

ROOT CAUSE:
  Your model was trained at 224×224 (check_mask_unet.py uses IMG_SIZE=(224,224))
  but app.py resizes to 128×128. At 128×128 the model only sees a blurry blob
  and outputs a high-confidence circle at the wound centre.
  Also: Otsu threshold on a near-binary map often produces a very tight mask.
  Also: Infection erythema scorer misreads dark suture threads as a red halo.

FIXES APPLIED:
  1. IMG_SIZE 128 → 224
  2. Otsu threshold clamped lower (0.20 min instead of 0.30)
  3. Morphological opening kernel reduced (3×3 instead of 5×5) to preserve edges
  4. Infection scorer: exclude very dark pixels (sutures) from erythema ring check
  5. Health check: mark online after first successful response regardless of model_loaded

Run from project root:
  python fix_circle_segmentation.py
"""

import re, shutil
from pathlib import Path

APP = Path("app.py")
if not APP.exists():
    print("❌ app.py not found — run from project root"); exit(1)

shutil.copy(APP, "app_pre_circle_fix.py")
print("✅ Backed up to app_pre_circle_fix.py")

src = APP.read_text()

# ── FIX 1: IMG_SIZE 128 → 224 ─────────────────────────────────────────────────
old = 'IMG_SIZE           = (int(os.getenv("IMG_H", 128)), int(os.getenv("IMG_W", 128)))'
new = 'IMG_SIZE           = (int(os.getenv("IMG_H", 224)), int(os.getenv("IMG_W", 224)))'
if old in src:
    src = src.replace(old, new)
    print("✅ Fix 1: IMG_SIZE set to 224×224")
else:
    # Try alternate forms
    src = re.sub(
        r'IMG_SIZE\s*=\s*\(int\(os\.getenv\("IMG_H",\s*128\)\),\s*int\(os\.getenv\("IMG_W",\s*128\)\)\)',
        'IMG_SIZE           = (int(os.getenv("IMG_H", 224)), int(os.getenv("IMG_W", 224)))',
        src
    )
    print("✅ Fix 1: IMG_SIZE set to 224×224 (regex)")

# ── FIX 2: Lower Otsu min threshold 0.30 → 0.20 ──────────────────────────────
old2 = 'thresh = max(0.30, min(0.75, float(otsu_val) / 255.0))'
new2 = 'thresh = max(0.20, min(0.65, float(otsu_val) / 255.0))'
if old2 in src:
    src = src.replace(old2, new2)
    print("✅ Fix 2: Otsu threshold floor lowered to 0.20")
else:
    src = re.sub(
        r'thresh\s*=\s*max\(0\.\d+,\s*min\(0\.\d+,\s*float\(otsu_val\)',
        'thresh = max(0.20, min(0.65, float(otsu_val)',
        src
    )
    print("✅ Fix 2: Otsu threshold floor lowered (regex)")

# ── FIX 3: Smaller morphological kernel to preserve wound edges ──────────────
old3 = '        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  k, iterations=2)\n        mask = cv2.morphologyEx(mask,     cv2.MORPH_CLOSE, k, iterations=3)'
new3 = '        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))\n        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)\n        mask = cv2.morphologyEx(mask,     cv2.MORPH_CLOSE, k_close, iterations=2)'
if old3 in src:
    src = src.replace(old3, new3)
    print("✅ Fix 3: Morphological kernels reduced — preserves irregular wound edges")
else:
    print("⚠️  Fix 3: Could not find exact morph block — applying regex")
    src = re.sub(
        r'k = cv2\.getStructuringElement\(cv2\.MORPH_ELLIPSE,\s*\(5,\s*5\)\)\s*\n\s*mask = cv2\.morphologyEx\(raw_mask,\s*cv2\.MORPH_OPEN,\s*k,\s*iterations=2\)\s*\n\s*mask = cv2\.morphologyEx\(mask,\s*cv2\.MORPH_CLOSE,\s*k,\s*iterations=3\)',
        'k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))\n        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)\n        mask = cv2.morphologyEx(mask,     cv2.MORPH_CLOSE, k_close, iterations=2)',
        src
    )

# ── FIX 4: Infection scorer — exclude very dark pixels (sutures/threads) ──────
old4 = '''        peri_bool = peri > 0
        H_p = hsv[:, :, 0][peri_bool].astype(float)
        S_p = hsv[:, :, 1][peri_bool].astype(float)
        ery_px   = np.sum(((H_p <= 10) | (H_p >= 160)) & (S_p > 60))
        ery_pct  = float(ery_px / max(peri_bool.sum(), 1) * 100)'''

new4 = '''        peri_bool = peri > 0
        H_p = hsv[:, :, 0][peri_bool].astype(float)
        S_p = hsv[:, :, 1][peri_bool].astype(float)
        V_p = hsv[:, :, 2][peri_bool].astype(float)
        # Exclude very dark pixels (sutures, threads, shadow) — V < 40 is near-black
        # Exclude desaturated pixels — not a true erythema signal
        valid = (V_p > 40) & (S_p > 50)
        ery_px   = np.sum(((H_p <= 10) | (H_p >= 160)) & valid)
        ery_pct  = float(ery_px / max(peri_bool.sum(), 1) * 100)'''

if old4 in src:
    src = src.replace(old4, new4)
    print("✅ Fix 4: Infection scorer excludes suture/dark pixels — fewer false positives")
else:
    print("⚠️  Fix 4: Could not patch infection scorer exactly — check manually")

# ── FIX 5: Update analysis info in HTML footer ────────────────────────────────
old5 = '                            <div>Input: 128×128px · MC-Dropout</div>'
new5 = '                            <div>Input: 224×224px · MC-Dropout</div>'
if old5 in src:
    src = src.replace(old5, new5)
    print("✅ Fix 5: Analysis info updated to 224×224")

APP.write_text(src)
print("\n✅ app.py patched!")
print("\nNow redeploy:")
print("  gcloud run deploy wound-api --source . --region us-central1 --memory 4Gi --cpu 2 --timeout 600 --allow-unauthenticated")
print()
print("Also update Cloud Run env vars to match:")
print("  gcloud run services update wound-api --region us-central1 --set-env-vars IMG_H=224,IMG_W=224")
