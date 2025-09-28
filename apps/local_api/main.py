# apps/local_api/main.py
from __future__ import annotations
import os, base64, time, requests
from typing import Optional, Dict, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:9000")

app = FastAPI(title="Wound Whisperer Local API", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class WoundSegResult(BaseModel):
    mask_area_px: int
    wound_percentage: float
    perimeter_px: float
    severity: str
    healing_potential: str
    mask_uri: Optional[str] = None

class AnalyzeRequest(BaseModel):
    image_url: Optional[HttpUrl] = Field(None)
    image_b64: Optional[str] = Field(None)
    request_id: Optional[str] = None
    threshold: float = 0.5

class AnalyzeResponse(BaseModel):
    id: str
    backend: str
    result: WoundSegResult
    mode: Literal["real","mock"]

REPORTS: Dict[str, Dict] = {}

def _to_b64(req: AnalyzeRequest) -> str:
    if req.image_b64:
        return req.image_b64
    if req.image_url:
        import urllib.request
        with urllib.request.urlopen(str(req.image_url), timeout=10) as r:
            return base64.b64encode(r.read()).decode()
    raise HTTPException(422, "Provide image_b64 or image_url")

@app.get("/healthz")
def healthz(): return {"status":"ok"}

@app.get("/readyz")
def readyz():
    try:
        j = requests.get(f"{MODEL_SERVER_URL}/readyz", timeout=3).json()
        return {"ready": bool(j.get("ready")), "backend":"simclr_unet", "model_server": MODEL_SERVER_URL}
    except Exception as e:
        return {"ready": False, "backend":"simclr_unet", "model_server": MODEL_SERVER_URL, "error": str(e)}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    rdy = requests.get(f"{MODEL_SERVER_URL}/readyz", timeout=3).json()
    if not rdy.get("ready"):
        raise HTTPException(503, "Model not ready (model server)")
    payload = {"image_b64": _to_b64(req), "threshold": float(req.threshold)}
    r = requests.post(f"{MODEL_SERVER_URL}/infer", json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(500, f"Model error: {r.text}")
    result = r.json()
    rid = req.request_id or str(int(time.time()*1000))
    REPORTS[rid] = {"created_at": int(time.time()), "backend": "simclr_unet", "result": result}
    return {"id": rid, "backend": "simclr_unet", "result": result, "mode":"real"}

@app.post("/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(file: UploadFile = File(...), threshold: float = 0.5):
    rdy = requests.get(f"{MODEL_SERVER_URL}/readyz", timeout=3).json()
    if not rdy.get("ready"):
        raise HTTPException(503, "Model not ready (model server)")
    payload = {"image_b64": base64.b64encode(await file.read()).decode(), "threshold": float(threshold)}
    r = requests.post(f"{MODEL_SERVER_URL}/infer", json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(500, f"Model error: {r.text}")
    result = r.json()
    rid = str(int(time.time()*1000))
    REPORTS[rid] = {"created_at": int(time.time()), "backend": "simclr_unet", "result": result}
    return {"id": rid, "backend": "simclr_unet", "result": result, "mode":"real"}

@app.get("/report/{id}")
def report(id: str):
    rep = REPORTS.get(id)
    if not rep: raise HTTPException(404, "Report not found")
    return rep

@app.get("/versionz")
def versionz():
    import fastapi, pydantic, requests
    return {
        "app": "proxy_api",
        "fastapi": fastapi.__version__,
        "pydantic": pydantic.__version__,
        "requests": requests.__version__,
    }