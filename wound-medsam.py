import logging
import sys
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import torch
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
import albumentations as A
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import segmentation_models as sm
import time
import base64
from PIL import Image
import io
from skimage import transform
from sklearn.metrics import precision_recall_fscore_support
from tenacity import retry, stop_after_attempt, wait_exponential
import gc
from tqdm.keras import TqdmCallback
from tensorflow.keras.utils import get_custom_objects
from segment_anything import build_sam_vit_b
from fpdf import FPDF
from diffusers import StableDiffusionPipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
import shap
import json
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout

# Enable mixed precision for TensorFlow
set_global_policy('mixed_float16')

# ---- Imports ----
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

def log_image_decision(log_path, image_name, brightness, contrast, enhanced, accepted, reason):
    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Image Name', 'Brightness', 'Contrast', 'Enhanced', 'Accepted', 'Reason'])
        writer.writerow([image_name, f"{brightness:.2f}", f"{contrast:.2f}", enhanced, accepted, reason])

# -----------------------------
# Integration inside load_data loop
# -----------------------------
# Place this inside the image loading loop in load_data()
# Make sure 'output' directory exists
os.makedirs('output', exist_ok=True)
log_path = 'output/image_decision_log.csv'

# ---- Define Loss First ----
class FocalTverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.7, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        tp = K.sum(y_true * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        fp = K.sum((1 - y_true) * y_pred)
        tversky = (tp + 1e-7) / (tp + self.alpha * fn + (1 - self.alpha) * fp + 1e-7)
        return K.pow(1 - tversky, self.gamma)

# ---- Register Custom Loss ----
def register_segmentation_models_custom_objects():
    from segmentation_models import metrics
    custom_objects = {
        'FocalTverskyLoss': FocalTverskyLoss(),
        'iou_score': metrics.IOUScore()
    }
    get_custom_objects().update(custom_objects)

# ---- Then call it ----
register_segmentation_models_custom_objects()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global MedSAM model
medsam_model = None
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
logger.info(f"Using device: {device}")

# Optimize TensorFlow for CPU
tf.config.set_soft_device_placement(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

def load_medsam_model(checkpoint_path):
    global medsam_model
    try:
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        medsam_model = build_sam_vit_b()
        medsam_model.load_state_dict(state_dict)
        # Skip quantization on MPS
        if torch.backends.mps.is_available():
            logger.info("Running on MPS, skipping quantization")
        else:
            medsam_model = torch.quantization.quantize_dynamic(medsam_model, {torch.nn.Linear}, dtype=torch.qint8)

        medsam_model.to(device)
        medsam_model.eval()
        logger.info(f"Loaded and quantized MedSAM model from {checkpoint_path}")
    except FileNotFoundError:
        logger.error(f"Checkpoint file {checkpoint_path} not found")
        medsam_model = None
    except Exception as e:
        logger.error(f"Failed to load MedSAM model: {str(e)}")
        medsam_model = None

def fine_tune_medsam(X_train, y_train, checkpoint_path, epochs=5):
    global medsam_model
    if medsam_model is None:
        load_medsam_model(checkpoint_path)
    optimizer = torch.optim.Adam(medsam_model.parameters(), lr=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    medsam_model.train()
    for epoch in range(epochs):
        for img, mask in zip(X_train, y_train):
            img = cv2.resize(img, (1024, 1024))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
            optimizer.zero_grad()
            image_embedding = medsam_model.image_encoder(img_tensor)
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(points=None, boxes=None, masks=None)
            outputs, _ = medsam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            loss = criterion(outputs, mask_tensor)
            loss.backward()
            optimizer.step()
        logger.info(f"MedSAM Fine-Tuning Epoch {epoch+1}, Loss: {loss.item()}")
    torch.save(medsam_model.state_dict(), "./models/best_medsam_model.pth")
    medsam_model.eval()

class VisualizationCallback(Callback):
    def __init__(self, X_val_resized, y_val_resized, save_dir="val_predictions"):
        super().__init__()
        num_samples = min(5, len(X_val_resized))
        self.X_val = X_val_resized[:num_samples]
        self.y_val = y_val_resized[:num_samples]
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if len(self.X_val) == 0:
            logger.warning("No validation samples available for visualization")
            return
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_binary = (y_pred > 0.5).astype(np.uint8)
        plt.figure(figsize=(10, 5))
        for i in range(len(self.X_val)):
            plt.subplot(2, 5, i + 1)
            plt.imshow(self.y_val[i].squeeze(), cmap='gray')
            plt.title(f'True Mask {i}')
            plt.axis('off')
            plt.subplot(2, 5, i + 6)
            plt.imshow(y_pred_binary[i].squeeze(), cmap='gray')
            plt.title(f'Pred Mask {i}')
            plt.axis('off')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch + 1}_{timestamp}.png'), dpi=150)
        plt.close('all')
        gc.collect()

def advanced_augmentation(is_training=True):
    transforms = [
        A.Resize(128, 128),
        A.Rotate(limit=40, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3) if is_training else A.NoOp(),
    ]
    return A.Compose(transforms, additional_targets={'mask': 'mask'})

def generate_synthetic_wounds(num_samples=100, output_dir="synthetic_wounds"):
    os.makedirs(output_dir, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(device)
    for i in range(num_samples):
        image = pipe("A realistic medical wound on skin, high detail", num_inference_steps=50).images[0]
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)  # Placeholder mask
        image.save(os.path.join(output_dir, f"synthetic_{i}.png"))
        cv2.imwrite(os.path.join(output_dir, f"synthetic_mask_{i}.png"), mask)
    return output_dir

def clean_data(images, masks):
    features = np.array([img.mean() for img in images]).reshape(-1, 1)
    clf = IsolationForest(contamination=0.1, random_state=42)
    outliers = clf.fit_predict(features)
    valid_indices = np.where(outliers == 1)[0]
    return images[valid_indices], masks[valid_indices]

def active_learning(model, X_unlabeled, num_samples=10):
    preds = model.predict(X_unlabeled, verbose=0)
    entropy = -np.sum(preds * np.log(preds + 1e-10), axis=-1).mean(axis=(1, 2))
    top_indices = np.argsort(entropy)[-num_samples:]
    return X_unlabeled[top_indices]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def check_image_quality(image):
    try:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        is_valid = brightness > 50 and brightness < 200 and contrast > 30
        message = "Image passed heuristic quality check" if is_valid else f"Image failed: brightness={brightness:.1f}, contrast={contrast:.1f}"
        return {"is_valid": is_valid, "message": message}
    except Exception as e:
        logger.error(f"Image quality check failed: {str(e)}")
        return {"is_valid": False, "message": f"Error in image quality check: {str(e)}"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def validate_segmentation(image, pred_mask):
    try:
        iou = sm.metrics.IOUScore()(np.expand_dims(pred_mask, 0), np.expand_dims(image, 0) if len(image.shape) == 2 else np.expand_dims(image, -1))
        is_valid = iou > 0.5
        message = f"IoU: {iou:.4f}, {'Valid segmentation' if is_valid else 'Invalid segmentation, consider retraining or adjusting MedSAM'}"
        return {"is_valid": is_valid, "message": message}
    except Exception as e:
        logger.error(f"Segmentation validation failed: {str(e)}")
        return {"is_valid": False, "message": f"Error in segmentation validation: {str(e)}"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_clinical_report(image, pred_mask, severity, healing_potential, validation_result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        wound_area = np.sum(pred_mask > 0)
        report_text = f"""
        Wound Assessment Report
        - Wound Description: Area {wound_area} pixels, {'mild' if severity.lower() == 'mild' else 'moderate' if severity.lower() == 'moderate' else 'severe'} severity.
        - Appearance: Based on area and segmentation, {'no obvious infection' if wound_area < 15000 else 'possible infection signs'} detected.
        - Clinical Recommendations: {'Regular dressing changes' if wound_area < 15000 else 'Dressing changes and possible antibiotics'}.
        - Patient Instructions: Keep wound clean, follow up in {'1 week' if wound_area < 15000 else '3 days'}.
        - Validation: {validation_result['message']}
        - Healing Potential: {healing_potential}
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Wound Assessment Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.multi_cell(0, 10, report_text)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"clinical_report_{timestamp}.pdf")
        pdf.output(report_path)
        return report_path
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def predict_healing_potential(mask, image, patient_metadata=None):
    wound_area = np.sum(mask > 0)
    if wound_area < 5000:
        severity = "Mild"
        healing_potential = "High (80%)"
    elif wound_area < 15000:
        severity = "Moderate"
        healing_potential = "Moderate (50%)"
    else:
        severity = "Severe"
        healing_potential = "Low (20%)"
    if patient_metadata and "diabetes" in patient_metadata and patient_metadata["diabetes"]:
        healing_potential = f"{float(healing_potential.split('(')[1].rstrip('%)')) - 20}%"
    return severity, healing_potential

def load_data(image_dir, mask_dir=None, img_size=(128, 128), api_key=None, is_training=True, mode="strict_qa"):
    from albumentations import Compose, Resize, Rotate, HorizontalFlip, RandomBrightnessContrast, NoOp
    import albumentations as A

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {image_dir}, is_training={is_training}, mode={mode}")

    images, masks, original_images = [], [], []
    if not os.path.exists(image_dir):
        logger.error(f"Image directory {image_dir} does not exist")
        return np.array([]), np.array([]), []

    os.makedirs('output', exist_ok=True)
    log_path = 'output/image_decision_log.csv'

    image_files = os.listdir(image_dir)
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        enhanced = False

        if mode in ["auto_enhanced", "strict_qa"] and brightness < 100:
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
            enhanced = True

        accepted = True
        reason = "Passed"

        if mode == "strict_qa" and api_key:
            quality_result = check_image_quality(img / 255.0, api_key)
            accepted = quality_result['is_valid']
            reason = quality_result['message']
            if not accepted:
                log_image_decision(log_path, img_name, brightness, contrast, enhanced, accepted, reason)
                continue

        log_image_decision(log_path, img_name, brightness, contrast, enhanced, accepted, reason)
        original_images.append(img.copy())

        # Load mask or fallback
        if mask_dir and os.path.exists(os.path.join(mask_dir, img_name)):
            mask = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                mask = (mask > 128).astype(np.uint8)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        images.append(img)
        masks.append(mask)

    if not images:
        logger.error("No valid images loaded")
        return np.array([]), np.array([]), []

    transform = Compose([
        Resize(img_size[0], img_size[1]),
        Rotate(limit=40, p=0.5) if is_training else NoOp(),
        HorizontalFlip(p=0.5) if is_training else NoOp(),
        RandomBrightnessContrast(p=0.3) if is_training else NoOp(),
    ])

    augmented_images, augmented_masks = [], []
    for img, mask in zip(images, masks):
        augmented = transform(image=img, mask=mask)
        augmented_images.append(augmented['image'] / 255.0)
        augmented_masks.append((augmented['mask'] > 0.5).astype(np.uint8))

    logger.info(f"Completed data processing for {len(augmented_images)} images")
    return np.array(augmented_images), np.array(augmented_masks), original_images

def enhance_if_dark(img, brightness_threshold=100, alpha=1.2, beta=20):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) < brightness_threshold:
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def load_data_with_medsam_fallback(image_dir, mask_dir=None, img_size=(128, 128), is_training=True, medsam_model_path=None):
    images, masks, sources, original_images = [], [], [], []
    if not os.path.exists(image_dir):
        logger.error(f"Image directory {image_dir} does not exist")
        return np.array([]), np.array([]), [], []
    transform = advanced_augmentation(is_training)
    image_files = os.listdir(image_dir)
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        quality_result = check_image_quality(img / 255.0)
        if not quality_result["is_valid"]:
            logger.warning(f"Skipping image {img_name}: {quality_result['message']}")
            continue
        original_images.append(img.copy())
        mask_path = os.path.join(mask_dir, img_name) if mask_dir else None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 128).astype(np.uint8)
                source = "manual"
            else:
                if medsam_model_path and medsam_model:
                    mask = medsam_segment(img, medsam_model_path)
                    source = "medsam" if mask is not None else "dummy"
                else:
                    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    source = "dummy"
        else:
            if medsam_model_path and medsam_model:
                mask = medsam_segment(img, medsam_model_path)
                source = "medsam" if mask is not None else "dummy"
            else:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                source = "dummy"
        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            source = "dummy"
        augmented = transform(image=img, mask=mask)
        images.append(augmented['image'] / 255.0)
        masks.append((augmented['mask'] > 0.5).astype(np.uint8))
        sources.append(source)
    images, masks = clean_data(np.array(images), np.array(masks))
    return np.array(images), np.array(masks), sources, original_images

def build_unet(input_shape=(128, 128, 3), dropout_rate=0.1):
    inputs = tf.keras.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(dropout_rate)(conv4, training=True)
    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop4)
    up5 = tf.keras.layers.Concatenate()([up5, conv3])
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = tf.keras.layers.Concatenate()([up6, conv2])
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = tf.keras.layers.Concatenate()([up7, conv1])
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_ensemble_unet(input_shape=(128, 128, 3)):
    model1 = build_unet(input_shape)
    model2 = sm.Unet('resnet34', input_shape=input_shape, classes=1, activation='sigmoid')
    return [model1, model2]

def create_dataset(X, y, batch_size, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def medsam_segment(image, medsam_model_path):
    global medsam_model
    if medsam_model is None:
        logger.error("MedSAM model not loaded")
        return None
    try:
        img_1024 = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img_1024 = cv2.cvtColor(img_1024, cv2.COLOR_RGB2BGR)
        img_tensor = torch.from_numpy(img_1024).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        box = np.array([[256, 256, 768, 768]])  # Placeholder learned prompt
        box_tensor = torch.from_numpy(box).float().to(device)
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_tensor)
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
            low_res_masks, _ = medsam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            mask = torch.sigmoid(low_res_masks).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        torch.cuda.empty_cache()
        return mask
    except Exception as e:
        logger.error(f"Error in MedSAM segmentation: {str(e)}")
        return None

def feature_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return canny / 255.0, sobelx / np.max(np.abs(sobelx)), sobely / np.max(np.abs(sobely))

def explainable_ai(model, image):
    model_grad = tf.keras.Model(model.inputs, [model.get_layer(index=-2).output, model.output])
    with tf.GradientTape() as tape:
        image_tensor = tf.expand_dims(image, axis=0)
        tape.watch(image_tensor)
        conv_outputs, predictions = model_grad(image_tensor)
        loss = predictions[:, :, :, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return cv2.resize(heatmap, (image.shape[1], image.shape[0]))

def explain_with_shap(model, image):
    explainer = shap.DeepExplainer(model, np.expand_dims(image, axis=0))
    shap_values = explainer.shap_values(np.expand_dims(image, axis=0))
    return shap_values

def combine_masks(mask1, mask2, method='union'):
    if method == 'union':
        return np.logical_or(mask1, mask2).astype(np.uint8)
    elif method == 'intersection':
        return np.logical_and(mask1, mask2).astype(np.uint8)
    elif method == 'average':
        combined = (mask1.astype(float) * 0.6 + mask2.astype(float) * 0.4)
        return (combined > 0.5).astype(np.uint8)
    else:
        raise ValueError(f"Unknown combination method: {method}")

def test_time_augmentation(model, image):
    augmentations = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: A.Rotate(limit=90, p=1)(image=x)['image']
    ]
    predictions = []
    for aug in augmentations:
        aug_img = aug(image)
        pred = model.predict(np.expand_dims(aug_img, axis=0), verbose=0)
        if aug == augmentations[1]:
            pred = np.fliplr(pred)
        elif aug == augmentations[2]:
            pred = np.flipud(pred)
        elif aug == augmentations[3]:
            pred = A.Rotate(limit=-90, p=1)(image=pred[0])['image']
            pred = np.expand_dims(pred, axis=0)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

def estimate_uncertainty(model, image, num_samples=10):
    predictions = []
    for _ in range(num_samples):
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predictions.append(pred)
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.var(predictions, axis=0)
    return mean_pred, uncertainty

def adversarial_training(model, X_train, y_train, epsilon=0.1):
    X_adv = X_train + epsilon * np.sign(np.random.randn(*X_train.shape))
    X_adv = np.clip(X_adv, 0, 1)
    train_dataset = create_dataset(np.concatenate([X_train, X_adv]), np.concatenate([y_train, y_train]), batch_size=4)
    model.fit(train_dataset, epochs=5, callbacks=[TqdmCallback(verbose=1)])
    return model

def calibrate_model(model, X_val, y_val):
    y_pred = model.predict(X_val, verbose=0).flatten()
    y_true = y_val.flatten()
    calibrator = CalibratedClassifierCV(base_estimator=None, method='sigmoid', cv='prefit')
    calibrator.fit(y_pred.reshape(-1, 1), y_true)
    return calibrator

def visualize_results(img, true_mask, pred_mask, orig_img, heatmap, canny, idx, output_dir, medsam_pred=None, validation_result=None, severity=None, healing_potential=None):
    val_dir = os.path.join(output_dir, "val_predictions")
    os.makedirs(val_dir, exist_ok=True)
    plt.figure(figsize=(20, 12))
    plt.subplot(3, 4, 1), plt.imshow(img), plt.title('Input Image'), plt.axis('off')
    plt.subplot(3, 4, 2), plt.imshow(true_mask, cmap='gray'), plt.title('True Mask'), plt.axis('off')
    plt.subplot(3, 4, 3), plt.imshow(pred_mask, cmap='gray'), plt.title('Hybrid Predicted Mask'), plt.axis('off')
    plt.subplot(3, 4, 4), plt.imshow(orig_img), plt.title('Original Image'), plt.axis('off')
    plt.subplot(3, 4, 5), plt.imshow(heatmap, cmap='jet'), plt.title('Grad-CAM Heatmap'), plt.axis('off')
    plt.subplot(3, 4, 6), plt.imshow(canny, cmap='gray'), plt.title('Canny Edges'), plt.axis('off')
    if medsam_pred is not None:
        plt.subplot(3, 4, 8), plt.imshow(medsam_pred, cmap='gray'), plt.title('MedSAM Predicted Mask'), plt.axis('off')
    if validation_result:
        plt.subplot(3, 4, 9), plt.text(0.5, 0.5, validation_result['message'][:200], wrap=True, ha='center', va='center'), plt.title('Validation Result'), plt.axis('off')
    if severity and healing_potential:
        plt.subplot(3, 4, 10), plt.text(0.5, 0.5, f"Severity: {severity}\nHealing: {healing_potential}", ha='center', va='center'), plt.title('Assessment'), plt.axis('off')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(val_dir, f'result_{idx}_{timestamp}.png'), dpi=150)
    plt.close('all')

def train_model(X_train, y_train, X_val, y_val, img_size=(128, 128), batch_size=4, epochs=50, model_save_path=None, pretrained_model_path=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
        model = build_unet(input_shape=(128, 128, 3))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss=FocalTverskyLoss(), metrics=[sm.metrics.IOUScore()])
        if pretrained_model_path:
            model.load_weights(pretrained_model_path)
        train_dataset = create_dataset(X_train_fold, y_train_fold, batch_size, is_training=True)
        val_dataset = create_dataset(X_val_fold, y_val_fold, batch_size, is_training=False)
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(f"{model_save_path}_fold{fold}", save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            VisualizationCallback(X_val_fold, y_val_fold),
            TqdmCallback(verbose=1)
        ]
        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)
        models.append(model)
    return models, None

def evaluate_model(model, X_test, y_test):
    X_test_resized = tf.image.resize(X_test, [128, 128], method='bilinear').numpy()
    y_test_resized = tf.image.resize(y_test, [128, 128], method='nearest').numpy()
    y_pred = model.predict(X_test_resized, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(np.uint8)
    y_test_flat = y_test_resized.flatten()
    y_pred_flat = y_pred_binary.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_flat, y_pred_flat, average='binary')
    iou = sm.metrics.IOUScore()(y_test_resized, y_pred_binary)
    logger.info(f"Test Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")

def evaluate_and_visualize(models, X_test, y_test, original_images, use_medsam=False, medsam_model_path=None, output_dir="output", hybrid_mode=False, combination_method='union', patient_metadata=None):
    val_dir = os.path.join(output_dir, "val_predictions")
    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    error_log = []
    calibrator = calibrate_model(models[0], X_test, y_test)
    batch_size = 4
    for start_idx in range(0, len(X_test), batch_size):
        end_idx = min(start_idx + batch_size, len(X_test))
        batch_imgs = X_test[start_idx:end_idx]
        batch_true_masks = y_test[start_idx:end_idx]
        batch_original_imgs = original_images[start_idx:end_idx]
        batch_imgs_resized = tf.image.resize(batch_imgs, [128, 128], method='bilinear').numpy()
        batch_preds = []
        for model in models:
            batch_preds.append(test_time_augmentation(model, batch_imgs_resized))
        batch_preds = np.mean(batch_preds, axis=0)
        batch_preds_calibrated = calibrator.predict_proba(batch_preds.flatten().reshape(-1, 1))[:, 1].reshape(batch_preds.shape)
        batch_preds = tf.image.resize(batch_preds_calibrated, [X_test.shape[1], X_test.shape[2]], method='nearest').numpy()
        for i, (img, true_mask, pred, orig_img) in enumerate(zip(batch_imgs, batch_true_masks, batch_preds, batch_original_imgs)):
            idx = start_idx + i
            pred_mask = (pred > 0.5).astype(np.uint8)
            if pred_mask.shape[-1] == 1:
                pred_mask = pred_mask.squeeze()
            iou = sm.metrics.IOUScore()(np.expand_dims(true_mask, 0), np.expand_dims(pred_mask, 0))
            if iou < 0.5:
                error_log.append({"idx": idx, "iou": float(iou), "image": f"image_{idx}"})
            medsam_pred = None
            if use_medsam and medsam_model_path:
                medsam_pred = medsam_segment(orig_img, medsam_model_path)
            if hybrid_mode and medsam_pred is not None:
                hybrid_mask = combine_masks(pred_mask, medsam_pred, method=combination_method)
            else:
                hybrid_mask = pred_mask
            canny, sobelx, sobely = feature_extraction(orig_img)
            heatmap = explainable_ai(models[0], img) if models else np.zeros_like(hybrid_mask)
            shap_values = explain_with_shap(models[0], img)
            severity, healing_potential = predict_healing_potential(hybrid_mask, orig_img, patient_metadata)
            validation_result = validate_segmentation(true_mask, hybrid_mask)
            report_path = generate_clinical_report(orig_img, hybrid_mask, severity, healing_potential, validation_result, report_dir)
            mean_pred, uncertainty = estimate_uncertainty(models[0], img)
            visualize_results(img, true_mask, hybrid_mask, orig_img, heatmap, canny, idx, output_dir, medsam_pred, validation_result, severity, healing_potential)
            gc.collect()
    with open(os.path.join(output_dir, "error_log.json"), "w") as f:
        json.dump(error_log, f)

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate wound segmentation model.')
    parser.add_argument('--train_image_dir', help='Path to training images')
    parser.add_argument('--train_mask_dir', help='Path to training masks')
    parser.add_argument('--test_image_dir', required=True, help='Path to test images')
    parser.add_argument('--test_mask_dir', help='Path to test masks')
    parser.add_argument('--img_size', type=int, default=128, help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--use_medsam', action='store_true', help='Use MedSAM for segmentation during evaluation')
    parser.add_argument('--use_medsam_fallback', action='store_true', help='Use MedSAM to generate masks for images without manual masks')
    parser.add_argument('--hybrid_mode', action='store_true', help='Combine U-Net and MedSAM masks')
    parser.add_argument('--combination_method', default='union', help='Combination method: union, intersection, average')
    parser.add_argument('--medsam_model_path', help='Path to MedSAM model weights')
    parser.add_argument('--pretrained_model_path', help='Path to pre-trained U-Net model weights')
    parser.add_argument('--model_save_path', default='best_unet_wound_model.keras', help='Path to save model weights')
    parser.add_argument('--output_dir', default='output', help='Base output directory')
    parser.add_argument('--patient_metadata', type=str, default='{"diabetes": false, "age": 0}', help='Patient metadata as JSON string')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument(
    '--image_mode',
    choices=['raw', 'auto_enhanced', 'strict_qa'],
    default='strict_qa',
    help='Image processing mode: raw (no enhancement), auto_enhanced (enhance dark images), strict_qa (enhance + quality check)'
)
    args = parser.parse_args()
    
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.debug(f"Arguments parsed: {vars(args)}")
    
    if not args.test_image_dir:
        logger.error("Test image directory is required")
        sys.exit(1)
    if args.use_medsam and not args.medsam_model_path:
        logger.error("MedSAM model path required when use_medsam is enabled")
        sys.exit(1)
    if args.hybrid_mode and not args.use_medsam:
        logger.error("Hybrid mode requires use_medsam to be enabled")
        sys.exit(1)
    
    import json
    patient_metadata = json.loads(args.patient_metadata)
    
    if args.use_medsam and args.medsam_model_path:
        load_medsam_model(args.medsam_model_path)
        if args.train_image_dir and args.train_mask_dir:
            X_train, y_train, _ = load_data(args.train_image_dir, args.train_mask_dir, img_size=(args.img_size, args.img_size), is_training=True)
            fine_tune_medsam(X_train, y_train, args.medsam_model_path)
    
    if args.use_medsam_fallback:
        X_test, y_test, sources_test, original_test_images = load_data_with_medsam_fallback(
            args.test_image_dir, args.test_mask_dir, img_size=(args.img_size, args.img_size),
            is_training=False, medsam_model_path=args.medsam_model_path
        )
    else:
        X_test, y_test, original_test_images = load_data(
            args.test_image_dir, args.test_mask_dir, img_size=(args.img_size, args.img_size),
            is_training=False
        )
        sources_test = []
    
    if len(X_test) == 0:
        logger.error("No test images loaded. Exiting.")
        sys.exit(1)
    
    y_test = np.expand_dims(y_test, axis=-1).astype('float32')
    X_test = X_test[:5]
    y_test = y_test[:5]
    original_test_images = original_test_images[:5]
    logger.info(f"Limited to {len(X_test)} test images for debugging")
    
    if args.pretrained_model_path:
        X_train, y_train = np.array([]), np.array([])
        X_val, y_val = np.array([]), np.array([])
    else:
        if not args.train_image_dir or not args.train_mask_dir:
            raise ValueError("Training directories required unless pre-trained model provided")
        X_train, y_train, _ = load_data(args.train_image_dir, args.train_mask_dir, img_size=(args.img_size, args.img_size), is_training=True)
        if len(X_train) == 0:
            logger.error("No training images loaded. Exiting.")
            sys.exit(1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        y_train = np.expand_dims(y_train, axis=-1).astype('float32')
        y_val = np.expand_dims(y_val, axis=-1).astype('float32')
    
    logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
    
    models, history = train_model(
        X_train, y_train, X_val, y_val,
        img_size=(args.img_size, args.img_size), batch_size=args.batch_size,
        epochs=args.epochs, model_save_path=args.model_save_path,
        pretrained_model_path=args.pretrained_model_path
    )
    
    for model in models:
        model = adversarial_training(model, X_train, y_train)
    
    evaluate_model(models[0], X_test, y_test)
    evaluate_and_visualize(
        models, X_test, y_test, original_test_images,
        use_medsam=args.use_medsam,
        medsam_model_path=args.medsam_model_path,
        output_dir=args.output_dir,
        hybrid_mode=args.hybrid_mode,
        combination_method=args.combination_method,
        patient_metadata=patient_metadata
    )
    logger.info(f"Evaluation completed. Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()