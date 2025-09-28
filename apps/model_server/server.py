# apps/model_server/server.py
from __future__ import annotations
import os, io, base64, logging, tempfile, shutil, traceback, sys, json
from typing import Optional, Any
import numpy as np
from PIL import Image

# ===== diagnostics =====
DIAG: dict[str, Any] = {
    "factory_entered": False,
    "model_path": None,
    "path_exists": None,
    "file_open_ok": None,
    "pid": None,
    "python": sys.version,
    "tf_version": None,
    "np_version": np.__version__,
    "tf_try": False,
    "tf_ok": None,
    "keras_try": False,
    "keras_ok": None,
    "converted_savedmodel": None,
    "warmup_try": False,
    "warmup_ok": None,
    "strict_warmup": bool(int(os.getenv("UNET_STRICT_WARMUP", "0"))),
    "disable_warmup": bool(int(os.getenv("UNET_DISABLE_WARMUP", "0"))),
}
LAST_ERROR: Optional[str] = None

# Fill with your custom objects if any
CUSTOM_OBJECTS: dict[str, Any] = {}

# Define Pydantic models at module level
from pydantic import BaseModel, Field

class InferRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 image (data URI ok)")
    threshold: float = 0.5

class InferResponse(BaseModel):
    mask_area_px: int
    wound_percentage: float
    perimeter_px: float
    severity: str
    healing_potential: str
    mask_uri: Optional[str] = None

def _set_ok():
    # clear any prior error on success
    global LAST_ERROR
    LAST_ERROR = None

def _set_err(prefix: str, e: Exception) -> str:
    global LAST_ERROR
    tb = f"{prefix}: {e}\n" + traceback.format_exc()
    LAST_ERROR = tb
    logging.getLogger("uvicorn").error(tb)
    return tb

def _warmup(model, img_size: tuple[int,int]):
    DIAG["warmup_try"] = True
    d = np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32)
    _ = model.predict(d, verbose=0)
    DIAG["warmup_ok"] = True

def load_unet_model(model_path: str, img_size: tuple[int,int]):
    """Return a loaded model using standalone Keras 3.x. Warm-up runs separately and may be non-fatal."""
    # Early checks
    DIAG["model_path"] = model_path
    DIAG["path_exists"] = os.path.exists(model_path)
    if not DIAG["path_exists"]:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        with open(model_path, "rb") as fh:
            fh.read(1)
        DIAG["file_open_ok"] = True
    except Exception as e:
        DIAG["file_open_ok"] = False
        raise

    # Use standalone Keras 3.x directly
    DIAG["keras_try"] = True
    try:
        import keras
        DIAG["tf_version"] = f"Keras {keras.__version__}"
        print(f"📦 Using Keras version: {keras.__version__}")
        
        # Load model with Keras 3.x
        m = keras.models.load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS or None)
        
        # Compile if needed
        if not getattr(m, "_is_compiled", False):
            m.compile(optimizer="adam", loss="binary_crossentropy")
        
        # Success bookkeeping
        DIAG["keras_ok"] = True
        _set_ok()
        print("✅ Model loaded successfully with Keras 3.x")
        return m
        
    except Exception as e_k:
        DIAG["keras_ok"] = False
        raise RuntimeError(_set_err("Keras 3.x load failed", e_k))

def build_app(model, img_size: tuple[int,int]):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import cv2
    from keras.utils import img_to_array

    app = FastAPI(title="WoundSeg Model Server", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])
    app.state.model = model
    app.state.img_size = img_size

    # Warm-up AFTER attaching model; non-fatal unless strict
    if app.state.model is not None and not DIAG["disable_warmup"]:
        try:
            _warmup(app.state.model, img_size)
        except Exception as e:
            _set_err("Warm-up failed", e)
            if DIAG["strict_warmup"]:
                app.state.model = None

    @app.get("/healthz")
    def healthz(): return {"status":"ok"}

    @app.get("/readyz")
    def readyz(): return {"ready": app.state.model is not None}

    @app.get("/last_errorz")
    def last_errorz():
        return {"last_error": LAST_ERROR}

    @app.get("/diagz")
    def diagz():
        return DIAG

    @app.get("/versionz")
    def versionz():
        import keras
        try:
            import tensorflow as tf
            tf_version = getattr(tf, "__version__", "unknown")
        except ImportError:
            tf_version = "not_installed"
        return {
            "app": "model_server",
            "keras": getattr(keras, "__version__", "unknown"),
            "tensorflow": tf_version,
            "numpy": DIAG["np_version"],
        }

    @app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest):
        if app.state.model is None:
            raise HTTPException(503, "Model not ready")
        b64 = req.image_b64.split(",",1)[1] if req.image_b64.startswith("data:") else req.image_b64
        try:
            img_bytes = base64.b64decode(b64, validate=True)
        except Exception:
            raise HTTPException(400, "Invalid base64 image")
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(app.state.img_size, Image.BILINEAR)
        arr = img_to_array(pil) / 255.0
        x = np.expand_dims(arr, 0).astype("float32")
        pred = app.state.model.predict(x, verbose=0)[0, :, :, 0]
        mask = (pred > req.threshold).astype("uint8") * 255

        area_px = int((mask > 0).sum())
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = float(cv2.arcLength(contours[0], True)) if contours else 0.0
        h, w = mask.shape
        pct = round(area_px / float(h * w), 6)

        buf = io.BytesIO()
        Image.fromarray(mask).save(buf, "PNG")
        mask_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        severity = "Mild" if pct < 0.01 else "Moderate" if pct < 0.05 else "Severe"
        healing = "Good" if pct < 0.01 else "Fair" if pct < 0.05 else "Poor"
        return {"mask_area_px": area_px, "wound_percentage": pct, "perimeter_px": perim,
                "severity": severity, "healing_potential": healing, "mask_uri": mask_uri}

    return app

def create_app():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("uvicorn")
    DIAG["factory_entered"] = True
    DIAG["pid"] = os.getpid()

    model_path = os.getenv("UNET_WEIGHTS_PATH", "/ABS/PATH/TO/simclr_unet_patch_wound.keras")
    img_h, img_w = map(int, os.getenv("UNET_IMG_SIZE", "128,128").split(","))
    log.info(f"📦 Loading model (factory): {model_path}")

    try:
        model = load_unet_model(model_path, (img_h, img_w))
        log.info("✅ Model deserialized (factory)")
    except Exception as e:
        _set_err("Factory load_unet_model failed", e)
        model = None

    try:
        app = build_app(model, (img_h, img_w))
        log.info(f"✅ App built; ready={app.state.model is not None} "
                 f"(tf_ok={DIAG.get('tf_ok')}, keras_ok={DIAG.get('keras_ok')}, "
                 f"warmup_ok={DIAG.get('warmup_ok')})")
        return app
    except Exception as e:
        _set_err("Factory build_app failed", e)
        # Build a tiny app that only exposes errors
        from fastapi import FastAPI
        tiny = FastAPI(title="WoundSeg Model Server (error)")
        @tiny.get("/readyz")
        def rz(): return {"ready": False}
        @tiny.get("/last_errorz")
        def lez(): return {"last_error": LAST_ERROR}
        @tiny.get("/diagz")
        def dz(): return DIAG
        return tiny

if __name__ == "__main__":
    import uvicorn, logging
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("MODEL_PORT","9100")), workers=1)