"""
fix_app.py
----------
Run this ONCE from your project folder:
    python fix_app.py

It makes two targeted fixes to app.py:
  1. Moves _init_db() inside an app context so Flask doesn't warn
  2. Replaces `import keras` with `import tensorflow as tf; keras = tf.keras`
     so it works even if the standalone keras package isn't installed
  3. Reads SIMCLR_MODEL_PATH from the environment at startup (not module import time)
"""

import re, shutil, sys
from pathlib import Path

APP = Path("app.py")

if not APP.exists():
    print("❌  app.py not found. Run this script from your project folder.")
    sys.exit(1)

# Back up first
shutil.copy(APP, "app_before_fix.py")
print("✅  Backed up to app_before_fix.py")

src = APP.read_text(encoding="utf-8")

# ── FIX 1 ─────────────────────────────────────────────────────────────────────
# Replace `import keras` / `keras.models.load_model` with tf.keras equivalent
# so it works whether keras is installed standalone or not
OLD_KERAS_IMPORT = "        import keras\n        logger.info(f\"[MODEL] Loading with Keras {keras.__version__}\")\n        MODEL = keras.models.load_model(MODEL_PATH, compile=False)"
NEW_KERAS_IMPORT = """        try:
            import keras as _keras
            logger.info(f"[MODEL] Using standalone Keras {_keras.__version__}")
            MODEL = _keras.models.load_model(MODEL_PATH, compile=False)
        except Exception:
            logger.info("[MODEL] Falling back to tf.keras")
            MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)"""

if OLD_KERAS_IMPORT in src:
    src = src.replace(OLD_KERAS_IMPORT, NEW_KERAS_IMPORT)
    print("✅  Fix 1 applied: keras import now has tf.keras fallback")
else:
    # Try alternative form present in some versions
    src = re.sub(
        r'import keras\s*\n\s*logger\.info\(f?"?\[MODEL\].*?keras.*?"\)\s*\n\s*MODEL = keras\.models\.load_model\(MODEL_PATH.*?\)',
        NEW_KERAS_IMPORT,
        src,
        flags=re.DOTALL
    )
    print("⚠️   Fix 1 applied via regex (check app.py manually if issues persist)")

# ── FIX 2 ─────────────────────────────────────────────────────────────────────
# Replace the bare module-level startup block with a proper with app.app_context() block
OLD_STARTUP = """logger.info("📦 Initialising database…")
try:
    _init_db()
    logger.info("✅ Database ready")
except Exception as e:
    logger.warning(f"DB init warning: {e}")

logger.info("📦 Loading model…")
try:
    load_model()
except Exception as e:
    logger.warning(f"Model load warning: {e}")

logger.info("🎉 WoundAI v3.0 ready")"""

NEW_STARTUP = """# ── Startup inside app context ──────────────────────────────────────────────
with app.app_context():
    logger.info("📦 Initialising database…")
    try:
        _init_db()
        logger.info("✅ Database ready")
    except Exception as e:
        logger.warning(f"DB init warning: {e}")

    logger.info("📦 Loading model…")
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Model load warning: {e}")

logger.info("🎉 WoundAI v3.0 ready")"""

if OLD_STARTUP in src:
    src = src.replace(OLD_STARTUP, NEW_STARTUP)
    print("✅  Fix 2 applied: startup now runs inside app.app_context()")
else:
    # Fuzzy fallback
    src = re.sub(
        r'logger\.info\("📦 Initialising database[^\n]*\n.*?logger\.info\("🎉 WoundAI v3\.0 ready"\)',
        NEW_STARTUP,
        src,
        flags=re.DOTALL
    )
    print("⚠️   Fix 2 applied via regex")

# ── FIX 3 ─────────────────────────────────────────────────────────────────────
# Make sure ensure_clean_model creates the directory first
OLD_ENSURE = "def ensure_clean_model(path: str):\n    if not os.path.exists(path):\n        return"
NEW_ENSURE = "def ensure_clean_model(path: str):\n    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n    if not os.path.exists(path):\n        return"

if OLD_ENSURE in src:
    src = src.replace(OLD_ENSURE, NEW_ENSURE)
    print("✅  Fix 3 applied: model directory auto-created")
else:
    print("⚠️   Fix 3 skipped (pattern not found — may already be correct)")

APP.write_text(src, encoding="utf-8")
print("\n✅  app.py patched successfully!")
print("\nNow run:")
print("  SIMCLR_MODEL_PATH=/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras python app.py")
