from __future__ import annotations
import os, io, base64, logging, tempfile, shutil, traceback, sys
from typing import Optional, Any, Tuple, Dict, List
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import json

# Diagnostics
DIAG: Dict[str, Any] = {
    "factory_entered": False,
    "pid": os.getpid(),
    "python": sys.version,
    "np_version": np.__version__,
    "model_path": None,
    "path_exists": None,
    "file_open_ok": None,
    "tf_version": None,
    "tf_try": False, "tf_ok": None,
    "keras_try": False, "keras_ok": None,
    "converted_savedmodel": None,
    "warmup_try": False, "warmup_ok": None,
    "strict_warmup": bool(int(os.getenv("UNET_STRICT_WARMUP", "0"))),
    "disable_warmup": bool(int(os.getenv("UNET_DISABLE_WARMUP", "0"))),
}
LAST_ERROR: Optional[str] = None

# Map Keras-3 "Custom>total_loss" to a harmless stub so deserialization works.
def _total_loss_stub(*args, **kwargs):
    return 0.0

CUSTOM_OBJECTS: Dict[str, Any] = {
    "Custom>total_loss": _total_loss_stub,
    "total_loss": _total_loss_stub,
}

def _set_err(prefix: str, e: Exception) -> str:
    global LAST_ERROR
    tb = f"{prefix}: {e}\n" + traceback.format_exc()
    LAST_ERROR = tb
    logging.getLogger("uvicorn").error(tb)
    return tb

def _set_ok():
    global LAST_ERROR
    LAST_ERROR = None

def load_unet_model(model_path: str, img_size: Tuple[int,int]):
    DIAG["model_path"] = model_path
    DIAG["path_exists"] = os.path.exists(model_path)
    if not DIAG["path_exists"]:
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        with open(model_path, "rb") as fh:
            fh.read(1)
        DIAG["file_open_ok"] = True
    except Exception:
        DIAG["file_open_ok"] = False
        raise

    # Try tf.keras first
    DIAG["tf_try"] = True
    try:
        import tensorflow as tf
        DIAG["tf_version"] = tf.__version__
        m = tf.keras.models.load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS or None)
        if not getattr(m, "_is_compiled", False):
            m.compile(optimizer="adam", loss="binary_crossentropy")
        DIAG["tf_ok"] = True
        _set_ok()
        return m
    except Exception as e_tf:
        DIAG["tf_ok"] = False
        _set_err("tf.keras load failed", e_tf)

    # Fallback: Keras 3 → SavedModel → tf.keras
    DIAG["keras_try"] = True
    try:
        import keras
        km = keras.models.load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS or None)
        DIAG["keras_ok"] = True
        _set_ok()
        return km
    except Exception as e_k:
        DIAG["keras_ok"] = False
        raise RuntimeError(_set_err("keras fallback failed", e_k))

def _warmup(model, img_size: Tuple[int,int]):
    DIAG["warmup_try"] = True
    d = np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32)
    _ = model.predict(d, verbose=0)
    DIAG["warmup_ok"] = True

def compute_gradcam(model, img_array, layer_name=None):
    """Compute Grad-CAM visualization for the model"""
    import tensorflow as tf
    
    # Get the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if hasattr(layer, 'filters') and len(layer.output_shape) == 4:
                layer_name = layer.name
                break
    
    if layer_name is None:
        return None
    
    # Create a model that outputs the last conv layer and final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_output = predictions[:, :, :, 0]  # Get the wound segmentation output
    
    # Compute gradients
    grads = tape.gradient(class_output, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps with the gradients
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[..., tf.newaxis]
    heatmap = conv_outputs @ pooled_grads
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy()

def compute_shap_values(model, img_array, background=None):
    """Compute SHAP values for model interpretability"""
    try:
        import shap
    except ImportError:
        return None
    
    if background is None:
        # Use zeros as background
        background = np.zeros((1, *img_array.shape[1:]))
    
    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(img_array)
    
    return shap_values[0] if isinstance(shap_values, list) else shap_values

def create_heatmap_visualization(mask, img_array, title="Wound Heatmap"):
    """Create a heatmap visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array[0])
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # Overlay
    overlay = img_array[0].copy()
    mask_colored = plt.cm.Reds(mask / mask.max())[:, :, :3]
    overlay = 0.6 * overlay + 0.4 * mask_colored
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return img_b64

def create_gradcam_visualization(gradcam, img_array, title="Grad-CAM"):
    """Create Grad-CAM visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    axes[0].imshow(img_array[0])
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Grad-CAM
    im = axes[1].imshow(gradcam, cmap='jet', alpha=0.8)
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return img_b64

def generate_pdf_report(img_array, mask, gradcam, metrics, report_path):
    """Generate a comprehensive PDF report"""
    with PdfPages(report_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.8, "Wound Analysis Report", fontsize=24, ha='center', weight='bold')
        ax.text(0.5, 0.7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=12, ha='center')
        ax.text(0.5, 0.6, f"Wound Area: {metrics['wound_area_mm2']:.2f} mm²", 
                fontsize=14, ha='center')
        ax.text(0.5, 0.5, f"Severity: {metrics['severity']}", 
                fontsize=14, ha='center')
        ax.text(0.5, 0.4, f"Healing Potential: {metrics['healing_potential']}", 
                fontsize=14, ha='center')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Analysis page
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Original image
        axes[0,0].imshow(img_array[0])
        axes[0,0].set_title("Original Image", fontsize=12)
        axes[0,0].axis('off')
        
        # Segmentation mask
        axes[0,1].imshow(mask, cmap='gray')
        axes[0,1].set_title("Segmentation Mask", fontsize=12)
        axes[0,1].axis('off')
        
        # Overlay
        overlay = img_array[0].copy()
        mask_colored = plt.cm.Reds(mask / mask.max())[:, :, :3]
        overlay = 0.6 * overlay + 0.4 * mask_colored
        axes[1,0].imshow(overlay)
        axes[1,0].set_title("Wound Overlay", fontsize=12)
        axes[1,0].axis('off')
        
        # Grad-CAM
        if gradcam is not None:
            im = axes[1,1].imshow(gradcam, cmap='jet', alpha=0.8)
            axes[1,1].set_title("Grad-CAM Analysis", fontsize=12)
            axes[1,1].axis('off')
            plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
        else:
            axes[1,1].text(0.5, 0.5, "Grad-CAM\nNot Available", ha='center', va='center')
            axes[1,1].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def build_app(model, img_size: Tuple[int,int]):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from fastapi.middleware.cors import CORSMiddleware
    from tensorflow.keras.utils import img_to_array

    class InferRequest(BaseModel):
        image_b64: str = Field(..., description="Base64 image (data URI ok)")
        threshold: float = 0.5
        include_gradcam: bool = True
        include_shap: bool = False
        include_heatmap: bool = True
        generate_pdf: bool = False

    class InferResponse(BaseModel):
        mask_area_px: int
        wound_percentage: float
        perimeter_px: float
        severity: str
        healing_potential: str
        wound_area_mm2: float
        mask_uri: Optional[str] = None
        heatmap_uri: Optional[str] = None
        gradcam_uri: Optional[str] = None
        pdf_uri: Optional[str] = None
        metrics: Dict[str, Any]

    app = FastAPI(title="Enhanced WoundSeg Model Server", version="2.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])
    app.state.model = model
    app.state.img_size = img_size

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

    @app.get("/diagz")
    def diagz(): return DIAG

    @app.get("/last_errorz")
    def last_errorz(): return {"last_error": LAST_ERROR}

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
        img_array = np.expand_dims(arr, 0).astype("float32")
        
        # Model prediction
        pred = app.state.model.predict(img_array, verbose=0)[0, :, :, 0]
        mask = (pred > req.threshold).astype("uint8") * 255

        # Basic metrics
        area_px = int((mask > 0).sum())
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = float(cv2.arcLength(contours[0], True)) if contours else 0.0
        h, w = mask.shape
        pct = round(area_px / float(h * w), 6)
        
        # Wound area in mm² (assuming 0.1 mm per pixel)
        scale_mm_per_pixel = 0.1
        wound_area_mm2 = area_px * (scale_mm_per_pixel ** 2)
        
        severity = "Mild" if pct < 0.01 else "Moderate" if pct < 0.05 else "Severe"
        healing = "Good" if pct < 0.01 else "Fair" if pct < 0.05 else "Poor"

        # Generate visualizations
        mask_uri = None
        heatmap_uri = None
        gradcam_uri = None
        pdf_uri = None
        
        # Mask visualization
        buf = io.BytesIO()
        Image.fromarray(mask).save(buf, "PNG")
        mask_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        
        # Heatmap visualization
        if req.include_heatmap:
            heatmap_uri = create_heatmap_visualization(mask, img_array)
        
        # Grad-CAM
        gradcam = None
        if req.include_gradcam:
            try:
                gradcam = compute_gradcam(app.state.model, img_array)
                if gradcam is not None:
                    gradcam_uri = create_gradcam_visualization(gradcam, img_array)
            except Exception as e:
                logging.warning(f"Grad-CAM computation failed: {e}")
        
        # SHAP values
        shap_values = None
        if req.include_shap:
            try:
                shap_values = compute_shap_values(app.state.model, img_array)
            except Exception as e:
                logging.warning(f"SHAP computation failed: {e}")
        
        # PDF report
        if req.generate_pdf:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"/tmp/wound_report_{timestamp}.pdf"
                metrics = {
                    "wound_area_mm2": wound_area_mm2,
                    "severity": severity,
                    "healing_potential": healing
                }
                generate_pdf_report(img_array, mask, gradcam, metrics, report_path)
                
                # Convert PDF to base64
                with open(report_path, "rb") as f:
                    pdf_b64 = base64.b64encode(f.read()).decode()
                pdf_uri = "data:application/pdf;base64," + pdf_b64
                
                # Clean up
                os.remove(report_path)
            except Exception as e:
                logging.warning(f"PDF generation failed: {e}")
        
        # Comprehensive metrics
        metrics = {
            "wound_area_px": area_px,
            "wound_percentage": pct,
            "perimeter_px": perim,
            "wound_area_mm2": wound_area_mm2,
            "scale_mm_per_pixel": scale_mm_per_pixel,
            "image_dimensions": {"height": h, "width": w},
            "threshold_used": req.threshold,
            "model_confidence": float(np.max(pred)),
            "gradcam_available": gradcam is not None,
            "shap_available": shap_values is not None
        }

        return InferResponse(
            mask_area_px=area_px,
            wound_percentage=pct,
            perimeter_px=perim,
            severity=severity,
            healing_potential=healing,
            wound_area_mm2=wound_area_mm2,
            mask_uri=mask_uri,
            heatmap_uri=heatmap_uri,
            gradcam_uri=gradcam_uri,
            pdf_uri=pdf_uri,
            metrics=metrics
        )

    return app

def create_app():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("uvicorn")
    DIAG["factory_entered"] = True

    model_path = os.getenv("UNET_WEIGHTS_PATH", "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras")
    img_h, img_w = list(map(int, os.getenv("UNET_IMG_SIZE", "128,128").split(",")))
    log.info(f"📦 Loading enhanced model server: {model_path}")

    try:
        model = load_unet_model(model_path, (img_h, img_w))
        log.info("✅ Enhanced model deserialized")
    except Exception as e:
        _set_err("Factory load_unet_model failed", e)
        model = None

    try:
        app = build_app(model, (img_h, img_w))
        log.info(f"✅ Enhanced app built; ready={app.state.model is not None}")
        return app
    except Exception as e:
        _set_err("Factory build_app failed", e)
        from fastapi import FastAPI
        tiny = FastAPI(title="Enhanced WoundSeg Model Server (error)")
        @tiny.get("/readyz"):     def rz():   return {"ready": False}
        @tiny.get("/last_errorz"):def lez():  return {"last_error": LAST_ERROR}
        @tiny.get("/diagz"):      def dz():   return DIAG
        return tiny

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("MODEL_PORT","9100")), workers=1)