#!/usr/bin/env python3
"""
End-to-end smoke test for wound segmentation API
"""
import os, sys, time, base64, json, io
from typing import Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from PIL import Image

MODEL_SERVER = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:9100")
PROXY_API    = os.environ.get("PROXY_API_URL",    "http://127.0.0.1:8000")

def _get(url: str):
    try:
        with urlopen(url, timeout=3) as r:
            return r.read().decode("utf-8")
    except Exception as e:
        return None

def _ready(url: str) -> bool:
    s = _get(url + "/readyz")
    if not s: return False
    try:
        j = json.loads(s)
        return bool(j.get("ready"))
    except Exception:
        return False

def wait_ready(name: str, base_url: str, timeout_s: int = 45) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _ready(base_url):
            print(f"✅ {name} ready: {base_url}")
            return
        time.sleep(0.5)
    print(f"❌ {name} NOT ready in {timeout_s}s: {base_url}")
    diag = _get(base_url + "/diagz") or _get(base_url + "/last_errorz")
    if diag:
        print(f"🔎 {name} diag:\n{diag}")
    sys.exit(1)

def make_synthetic_png(size: Tuple[int,int]=(128,128)) -> bytes:
    """Simple RGB test image with a red square (wound-like) on green background."""
    w,h = size
    img = Image.new("RGB", size, (20,160,20))
    # draw a small red square
    for y in range(h//3, h//3 + h//6):
        for x in range(w//3, w//3 + w//6):
            img.putpixel((x,y), (200,30,30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def analyze_via_proxy(image_bytes: bytes):
    b64 = base64.b64encode(image_bytes).decode()
    payload = json.dumps({"image_b64": b64}).encode("utf-8")
    req = Request(PROXY_API + "/analyze", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as r:
            body = r.read().decode("utf-8")
            return r.status, json.loads(body)
    except HTTPError as e:
        print("❌ Proxy returned error:", e, e.read().decode("utf-8"))
        sys.exit(1)
    except URLError as e:
        print("❌ Proxy unreachable:", e)
        sys.exit(1)

def main():
    print(f"➡️  Checking model server: {MODEL_SERVER}")
    wait_ready("Model Server", MODEL_SERVER)

    print(f"➡️  Checking proxy API: {PROXY_API}")
    wait_ready("Proxy API", PROXY_API)

    print("➡️  Generating synthetic image…")
    img = make_synthetic_png()

    print("➡️  Calling /analyze via proxy…")
    status, resp = analyze_via_proxy(img)
    if status != 200:
        print("❌ Non-200 from proxy:", status, json.dumps(resp, indent=2))
        sys.exit(1)

    # Basic assertions
    mode = resp.get("mode")
    if mode != "real":
        print("❌ Expected mode='real', got:", mode)
        sys.exit(1)

    result = resp.get("result", {})
    pct = float(result.get("wound_percentage", 0.0))
    area = int(result.get("mask_area_px", 0))
    severity = result.get("severity")

    print("✅ Inference OK")
    print(json.dumps({
        "area_px": area,
        "wound_percentage": pct,
        "severity": severity,
    }, indent=2))

    # Sanity thresholds for synthetic image (non-zero but small)
    if not (area > 0 and 0.0 < pct < 0.2):
        print("⚠️  Unusual segmentation metrics for synthetic image — check thresholds.")
    sys.exit(0)

if __name__ == "__main__":
    main()