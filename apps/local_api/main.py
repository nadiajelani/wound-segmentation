# apps/local_api/main.py
from __future__ import annotations
import io, time, base64, logging
from typing import Dict, Optional, Literal

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

import numpy as np
from PIL import Image
import cv2

# ---- LOGGING ----
log = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

# ==============================
#  WOUNDSEG PROVIDER (Keras)
# ==============================
# Inference flow mirrors your scripts:
# - tf.keras.models.load_model(..., compile=False)
# - load_img/img_to_array + 1/255.0, target_size=(128,128)
# - model.predict(...) -> threshold at 0.5 to get binary mask
# - area/perimeter from mask using OpenCV

class WoundSegResult(BaseModel):
    mask_area_px: int
    wound_percentage: float
    perimeter_px: float
    condition: str
    healing_potential: str
    severity: str
    confidence: float = 0.85
    bbox_xywh: Optional[list[int]] = None
    mask_uri: Optional[str] = None
    notes: Optional[str] = None

class BaseWoundSegProvider:
    name: str = "base"
    _loaded: bool = False
    def load(self) -> None: ...
    def analyze_bytes(self, image_bytes: bytes) -> WoundSegResult: ...

class SimCLR_UNet_Provider(BaseWoundSegProvider):
    name = "simclr_unet"

    def __init__(self,
                 model_path: str = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras",
                 img_size: tuple[int, int] = (128, 128),
                 threshold: float = 0.5):
        self.model_path = model_path
        self.img_size = img_size
        self.threshold = threshold
        self._model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            # Lazy import TF to keep /healthz snappy
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self.model_path, compile=False)
            # Optional warm-up with a dummy tensor to confirm graph is ready
            import numpy as np
            dummy = np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.float32)
            _ = self._model.predict(dummy, verbose=0)
            self._loaded = True
            log.info("🔥 [simclr_unet] models loaded (warm-up complete)")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            # Fallback to mock mode
            self._loaded = True
            self._model = None
            log.warning("⚠️ Running in mock mode - model loading failed")

    def _preprocess(self, pil_img: Image.Image) -> np.ndarray:
        from tensorflow.keras.preprocessing.image import img_to_array
        img = pil_img.convert("RGB").resize(self.img_size, Image.BILINEAR)  # target size
        arr = img_to_array(img) / 255.0                                    # /255.0
        return np.expand_dims(arr, axis=0)

    def analyze_bytes(self, image_bytes: bytes) -> WoundSegResult:
        if not self._loaded:
            raise RuntimeError("Model not ready")

        # If model failed to load, return mock results
        if self._model is None:
            log.warning("Using mock analysis - model not available")
            return WoundSegResult(
                mask_area_px=12345,
                wound_percentage=0.05,
                perimeter_px=450.0,
                condition="Mock wound - testing mode",
                healing_potential="Unknown",
                severity="Test",
                bbox_xywh=[10, 10, 100, 100],
                mask_uri=None,
                notes="Mock analysis - model not loaded",
            )

        pil = Image.open(io.BytesIO(image_bytes))
        model_in = self._preprocess(pil)

        # ---- Predict & threshold to binary mask ----
        pred = self._model.predict(model_in, verbose=0)
        logits = pred[0, :, :, 0]
        mask = (logits > self.threshold).astype("uint8") * 255

        # ---- Metrics (area, perimeter) ----
        area_px = int(np.sum(mask > 0))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter_px = float(cv2.arcLength(contours[0], True)) if contours else 0.0

        # BBox
        ys, xs = np.where(mask > 0)
        bbox_xywh = None
        if ys.size and xs.size:
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            bbox_xywh = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]

        # Encode mask preview (data URI)
        buf = io.BytesIO()
        Image.fromarray(mask).save(buf, format="PNG")
        mask_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        # % of image area
        h, w = mask.shape[:2]
        wound_pct = round(area_px / float(h * w), 6)

        # Simple condition bucketing (you can replace with your rules)
        if wound_pct < 0.01:
            condition, healing_potential, severity = "Small wound - healing well", "Good", "Mild"
        elif wound_pct < 0.05:
            condition, healing_potential, severity = "Medium wound - stable", "Fair", "Moderate"
        else:
            condition, healing_potential, severity = "Large wound - needs attention", "Poor", "Severe"

        return WoundSegResult(
            mask_area_px=area_px,
            wound_percentage=wound_pct,
            perimeter_px=perimeter_px,
            condition=condition,
            healing_potential=healing_potential,
            severity=severity,
            bbox_xywh=bbox_xywh,
            mask_uri=mask_uri,
            notes="SimCLR-UNet inference via tf.keras; threshold=0.5",
        )

# ==============================
#  FASTAPI APP
# ==============================
app = FastAPI(
    title="Wound Whisperer Local API",
    version="0.3.0",
    description="HTTP endpoints exposing wound segmentation & reporting."
)

# CORS for SPA at Vite (5173) and React (3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

READY = False
REPORTS: Dict[str, Dict] = {}

class AnalyzeRequest(BaseModel):
    image_url: Optional[HttpUrl] = Field(None, description="Publicly reachable image URL")
    image_b64: Optional[str] = Field(None, description="Base64-encoded image (data URI or raw base64)")
    request_id: Optional[str] = None

class AnalyzeResponse(BaseModel):
    id: str
    backend: str
    result: WoundSegResult

# Choose active backend
provider: BaseWoundSegProvider = SimCLR_UNet_Provider(
    model_path="/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras",
    img_size=(128, 128),
    threshold=0.5,
)

@app.on_event("startup")
def startup_event():
    global READY
    provider.load()     # warm-up once
    READY = True

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"ready": READY, "backend": provider.name}

def _decode_from_request(req: AnalyzeRequest) -> bytes:
    if req.image_b64:
        b64 = req.image_b64
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            return base64.b64decode(b64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
    if req.image_url:
        import urllib.request
        try:
            with urllib.request.urlopen(str(req.image_url), timeout=10) as resp:
                return resp.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    raise HTTPException(status_code=422, detail="Provide image_b64 or image_url")

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not READY:
        raise HTTPException(status_code=503, detail="Model not ready")
    img_bytes = _decode_from_request(req)
    try:
        result = provider.analyze_bytes(img_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    report_id = req.request_id or str(int(time.time() * 1000))
    REPORTS[report_id] = {
        "created_at": int(time.time()),
        "backend": provider.name,
        "result": result.dict(),
    }
    return {"id": report_id, "backend": provider.name, "result": result}

@app.post("/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(file: UploadFile = File(...)):
    if not READY:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        img_bytes = await file.read()
        result = provider.analyze_bytes(img_bytes)
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    report_id = str(int(time.time() * 1000))
    REPORTS[report_id] = {
        "created_at": int(time.time()),
        "backend": provider.name,
        "result": result.dict(),
    }
    return {"id": report_id, "backend": provider.name, "result": result}

@app.get("/report/{id}")
def get_report(id: str):
    report = REPORTS.get(id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)