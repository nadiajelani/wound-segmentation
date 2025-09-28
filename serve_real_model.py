# serve_real_model.py
from __future__ import annotations
import os, io, base64, logging, traceback
from typing import Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# --- Config ---
MODEL_PATH   = os.getenv("UNET_WEIGHTS_PATH", "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras")
IMG_SIZE_STR = os.getenv("UNET_IMG_SIZE", "128,128")
IMG_SIZE: Tuple[int,int] = tuple(map(int, IMG_SIZE_STR.split(",")))  # (H,W)
THRESH       = float(os.getenv("UNET_THRESH", "0.5"))

# Map any serialized custom names used during training. Add more if needed.
def _total_loss_stub(*args, **kwargs): return 0.0
CUSTOM_OBJECTS: Dict[str, Any] = {
    "Custom>total_loss": _total_loss_stub,
    "total_loss": _total_loss_stub,
}

# --- Globals/diag ---
app = FastAPI(title="Real SimCLR-UNet Server (Keras3)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)
MODEL = None
DIAG: Dict[str, Any] = {"ready": False, "model_path": MODEL_PATH, "img_size": IMG_SIZE, "load_error": None}

class InferRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 of an RGB image. Data-URI ok.")
    threshold: float = Field(THRESH, ge=0.0, le=1.0)

class InferResponse(BaseModel):
    mask_area_px: int
    wound_percentage: float
    perimeter_px: float
    severity: str
    healing_potential: str
    mask_uri: Optional[str] = None

def _load_model():
    global MODEL
    try:
        import keras
        # Keras 3: allow loading with non-registered objects
        MODEL = keras.models.load_model(
            MODEL_PATH, compile=False, custom_objects=CUSTOM_OBJECTS or None, safe_mode=False
        )
        DIAG["ready"] = True
        DIAG["keras"] = getattr(__import__("keras"), "__version__", "unknown")
        # optional compile for predict graph
        try:
            MODEL.compile(optimizer="adam", loss="binary_crossentropy")
        except Exception:
            pass
    except Exception as e:
        DIAG["load_error"] = f"{e}\n{traceback.format_exc()}"
        logging.exception("Real model load failed")
        MODEL = None
        DIAG["ready"] = False

def _preprocess(b64: str) -> np.ndarray:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True))).convert("RGB")
    img = img.resize(IMG_SIZE[::-1], Image.BILINEAR)  # (W,H)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)  # (1,H,W,3)

def _postprocess(prob: np.ndarray, thr: float) -> InferResponse:
    import cv2
    prob = prob[0, :, :, 0]
    mask = (prob >= thr).astype("uint8") * 255
    h, w = mask.shape
    area = int((mask > 0).sum())
    # perimeter (largest contour)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(cv2.arcLength(max(contours, key=cv2.contourArea), True)) if contours else 0.0
    pct = round(area / float(h * w), 6)
    # small preview
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, "PNG")
    mask_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    severity = "Mild" if pct < 0.01 else "Moderate" if pct < 0.05 else "Severe"
    healing  = "Good" if pct < 0.01 else "Fair" if pct < 0.05 else "Poor"
    return InferResponse(
        mask_area_px=area, wound_percentage=pct, perimeter_px=perim,
        severity=severity, healing_potential=healing, mask_uri=mask_uri
    )

@app.on_event("startup")
def _startup():
    _load_model()

@app.get("/readyz")
def readyz(): return {"ready": bool(MODEL is not None), "model_path": MODEL_PATH, "diag": DIAG}

@app.get("/diagz")
def diagz(): return DIAG

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    x = _preprocess(req.image_b64)
    try:
        y = MODEL.predict(x, verbose=0)
    except Exception as e:
        DIAG["predict_error"] = f"{e}\n{traceback.format_exc()}"
        raise HTTPException(500, f"Predict failed: {e}")
    return _postprocess(y, req.threshold)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("REAL_PORT", "9101")), workers=1, log_level="info")