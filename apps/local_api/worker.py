# apps/local_api/worker.py
import io, os, base64
from PIL import Image
import numpy as np

def _load_model(model_path, img_size):
    import tensorflow as tf  # import only inside worker
    m = tf.keras.models.load_model(model_path, compile=False)
    if not getattr(m, "_is_compiled", False):
        m.compile(optimizer="adam", loss="binary_crossentropy")
    d = np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32)
    _ = m.predict(d, verbose=0)
    return m

def _infer(model, img_bytes, img_size, threshold=0.5):
    from tensorflow.keras.preprocessing.image import img_to_array
    import cv2
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(img_size, Image.BILINEAR)
    arr = img_to_array(pil) / 255.0
    x = np.expand_dims(arr, 0).astype("float32")
    pred = model.predict(x, verbose=0)[0, :, :, 0]
    mask = (pred > threshold).astype("uint8") * 255

    # metrics
    area_px = int((mask > 0).sum())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    h, w = mask.shape
    pct = round(area_px / float(h * w), 6)

    # preview
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, "PNG")
    mask_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    severity = "Mild" if pct < 0.01 else "Moderate" if pct < 0.05 else "Severe"
    healing = "Good" if pct < 0.01 else "Fair" if pct < 0.05 else "Poor"
    return {
        "mask_area_px": area_px,
        "wound_percentage": pct,
        "perimeter_px": perim,
        "severity": severity,
        "healing_potential": healing,
        "mask_uri": mask_uri,
    }

def worker_loop(shared, requests, results):
    model_path = shared["model_path"]
    img_size = tuple(shared.get("img_size", (128, 128)))
    shared["status"] = "loading"
    try:
        model = _load_model(model_path, img_size)
        shared["status"] = "ready"
    except Exception as e:
        shared["status"] = f"error: {e}"
        return

    while True:
        task = requests.get()  # blocking
        if task == "__shutdown__":
            shared["status"] = "stopped"
            return
        rid, img_bytes, threshold = task
        try:
            results[rid] = {"ok": True, "result": _infer(model, img_bytes, img_size, threshold)}
        except Exception as e:
            results[rid] = {"ok": False, "error": str(e)}