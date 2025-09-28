# apps/model_server/main.py
from __future__ import annotations
import io, os, base64, logging, asyncio
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.getenv("UNET_WEIGHTS_PATH", "/ABS/PATH/TO/simclr_unet_patch_wound.keras")
IMG_H, IMG_W = map(int, os.getenv("UNET_IMG_SIZE", "128,128").split(","))

class InferRequest(BaseModel):
    image_b64: str = Field(..., description="base64 image (may be data URI)")
    threshold: float = 0.5

class InferResponse(BaseModel):
    mask_area_px: int
    wound_percentage: float
    perimeter_px: float
    severity: str
    healing_potential: str
    mask_uri: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.ready = False
    try:
        import tensorflow as tf
        log.info(f"📦 Loading model: {MODEL_PATH}")
        m = tf.keras.models.load_model(MODEL_PATH, compile=False)
        if not getattr(m, "_is_compiled", False):
            m.compile(optimizer="adam", loss="binary_crossentropy")
        # warm-up
        d = np.zeros((1, IMG_H, IMG_W, 3), dtype=np.float32)
        _ = m.predict(d, verbose=0)
        app.state.model = m
        app.state.ready = True
        log.info("✅ Model ready")
    except Exception:
        log.exception("Model load failed")
        app.state.model = None
        app.state.ready = False
    try:
        yield
    finally:
        log.info("🧹 Model server shutdown")

app = FastAPI(title="WoundSeg Model Server", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/healthz")
def healthz(): return {"status": "ok"}

@app.get("/readyz")
def readyz(): return {"ready": bool(app.state.ready), "model_path": MODEL_PATH}

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if not app.state.ready or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    # decode
    b64 = req.image_b64.split(",", 1)[1] if req.image_b64.startswith("data:") else req.image_b64
    try:
        img_b = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")
    from tensorflow.keras.preprocessing.image import img_to_array
    import cv2

    pil = Image.open(io.BytesIO(img_b)).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
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