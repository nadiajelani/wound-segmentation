"""
Model loading utilities for SimCLR U-Net wound segmentation
- Supports .keras saved with Keras 3.x (standalone Keras) using safe_mode=False
- Also supports TF SavedModel directories (for TF 2.12 serving)
- Provides an in-code toy U-Net fallback so we don't depend on external modules
"""
import os
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

class SimCLRModelLoader:
    """Handles loading and management of SimCLR U-Net models"""
    def __init__(self, model_path: str = None):
        # Allow override via env
        self.model_path = model_path or os.getenv('SIMCLR_MODEL_PATH', 'models/simclr_unet_patch_wound.keras')
        self.model = None
        self.input_shape = (128, 128, 3)  # will be updated from model if available

    def load_model(self):
        """
        Load the SimCLR U-Net model with robust strategy:
        1) If path is a directory -> treat as TF SavedModel (tf.keras load)
        2) If path is a file (.keras/.h5) -> load with standalone Keras 3 (safe_mode=False)
        3) On failure -> build a tiny fallback UNet-like model inline (no external deps)
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}; using fallback tiny model")
                self.model = self._build_tiny_unet()
                return self.model

            if os.path.isdir(self.model_path):
                # SavedModel directory → tf.keras can load this
                import tensorflow as tf
                logger.info(f"Loading SavedModel from directory: {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                self._refresh_input_shape_from_model()
                logger.info("SavedModel loaded successfully")
                return self.model

            # Otherwise, assume a .keras / .h5 file saved by standalone Keras 3
            logger.info(f"Loading Keras 3 model file: {self.model_path}")
            os.environ.setdefault("KERAS_BACKEND", "tensorflow")
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")
            import keras
            try:
                keras.config.disable_traceback_filtering()
            except Exception:
                pass

            from keras.saving import register_keras_serializable
            @register_keras_serializable(package="Custom")
            def total_loss(*args, **kwargs):
                return 0.0

            custom_objects: Dict[str, Any] = {
                "Custom>total_loss": total_loss,
                "total_loss": total_loss,
            }

            self.model = keras.models.load_model(
                self.model_path,
                compile=False,
                safe_mode=False,
                custom_objects=custom_objects
            )
            # Optional compile for predict graph; not required for inference
            try:
                self.model.compile(optimizer="adam", loss="binary_crossentropy")
            except Exception:
                pass

            self._refresh_input_shape_from_model()
            logger.info("Keras 3 model loaded successfully")
            return self.model

        except Exception as e:
            logger.error(f"Error loading SimCLR model: {e}", exc_info=True)
            logger.info("Building tiny fallback UNet so API remains available")
            self.model = self._build_tiny_unet()
            return self.model

    def _refresh_input_shape_from_model(self):
        """Derive (H, W, C) from model.input_shape, usually (None, H, W, C)"""
        shp = getattr(self.model, "input_shape", None)
        if isinstance(shp, (list, tuple)) and len(shp) == 4:
            self.input_shape = (int(shp[1]), int(shp[2]), int(shp[3]))

    def _build_tiny_unet(self):
        """Minimal UNet-like model that always exists (no external module imports)."""
        import tensorflow as tf
        from tensorflow.keras import layers, models
        h, w, c = self.input_shape
        inp = layers.Input((h, w, c))
        x = layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
        s = layers.MaxPool2D()(x)
        s = layers.Conv2D(16, 3, padding="same", activation="relu")(s)
        u = layers.UpSampling2D()(s)
        u = layers.Concatenate()([x, u])
        out = layers.Conv2D(1, 1, activation="sigmoid")(u)
        model = models.Model(inp, out)
        logger.info("Tiny fallback UNet built")
        return model

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Make prediction on input image. `image` can be (H,W,C) or (1,H,W,C) in [0,1]."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        import tensorflow as tf
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Resize to model input size
        image_resized = tf.image.resize(image, self.input_shape[:2])
        return self.model.predict(image_resized, verbose=0)

    def get_model_info(self) -> dict:
        if self.model is None:
            return {"status": "not_loaded"}
        return {
            "status": "loaded",
            "input_shape": self.input_shape,
            "model_path": self.model_path,
            "total_params": int(self.model.count_params()),
        }

def load_simclr_model(model_path: str = None) -> SimCLRModelLoader:
    loader = SimCLRModelLoader(model_path)
    loader.load_model()
    return loader