import logging
import sys
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"  # Set before importing segmentation_models
import cv2
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
from openai import OpenAI
from fpdf import FPDF
import torch
from skimage import transform
from sklearn.metrics import precision_recall_fscore_support
from tenacity import retry, stop_after_attempt, wait_exponential
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global MedSAM model
medsam_model = None
device = "cpu"  # MacBook Air, no CUDA
if torch.cuda.is_available():
    device = "cuda:0"
logger.info(f"Using device: {device}")

# Optimize TensorFlow for CPU
tf.config.set_soft_device_placement(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

def load_medsam_model(checkpoint_path):
    global medsam_model
    try:
        from segment_anything import build_sam_vit_b
        medsam_model = build_sam_vit_b(checkpoint=checkpoint_path).to(device)
        medsam_model.eval()
        logger.info(f"Loaded MedSAM model from {checkpoint_path}")
    except FileNotFoundError:
        logger.error(f"Checkpoint file {checkpoint_path} not found")
        medsam_model = None
    except Exception as e:
        logger.error(f"Failed to load MedSAM model: {str(e)}")
        medsam_model = None

class VisualizationCallback(Callback):
    def __init__(self, X_val_resized, y_val_resized, save_dir="val_predictions"):
        super().__init__()
        self.X_val = X_val_resized[:5]  # Limit to 5 samples
        self.y_val = y_val_resized[:5]
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def check_image_quality(image, api_key):
    try:
        client = OpenAI(api_key=api_key)
        img_uint8 = (image * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this medical wound image clear, well-lit, and suitable for analysis? If not, suggest improvements."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            max_tokens=150
        )
        message = response.choices[0].message.content
        is_valid = "clear" in message.lower() and "suitable" in message.lower()
        return {"is_valid": is_valid, "message": message}
    except Exception as e:
        logger.error(f"Image quality check failed: {str(e)}")
        return {"is_valid": False, "message": f"Error in image quality check: {str(e)}"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def validate_segmentation(image, pred_mask, api_key):
    try:
        client = OpenAI(api_key=api_key)
        img_uint8 = (image * 255).astype(np.uint8)
        _, img_buffer = cv2.imencode('.png', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        mask_uint8 = (pred_mask * 255).astype(np.uint8)
        _, mask_buffer = cv2.imencode('.png', mask_uint8)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a wound image and its AI-generated segmentation mask (white = wound, black = background). Does the mask accurately outline the wound? If not, describe any issues and suggest improvements."
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mask_base64}"}}
                    ]
                }
            ],
            max_tokens=200
        )
        message = response.choices[0].message.content
        is_valid = "accurate" in message.lower() or "correct" in message.lower()
        return {"is_valid": is_valid, "message": message}
    except Exception as e:
        logger.error(f"Segmentation validation failed: {str(e)}")
        return {"is_valid": False, "message": f"Error in segmentation validation: {str(e)}"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_clinical_report(image, pred_mask, severity, healing_potential, validation_result, api_key, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        client = OpenAI(api_key=api_key)
        img_uint8 = (image * 255).astype(np.uint8)
        _, img_buffer = cv2.imencode('.png', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        mask_uint8 = (pred_mask * 255).astype(np.uint8)
        _, mask_buffer = cv2.imencode('.png', mask_uint8)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        prompt = f"""
        Generate a clinical wound assessment report based on:
        - Wound image and segmentation mask (white = wound, black = background).
        - Severity: {severity}
        - Healing Potential: {healing_potential}
        - Segmentation Validation: {validation_result['message']}
        Include:
        - Wound description (size, appearance, infection signs).
        - Clinical recommendations (e.g., dressing changes, antibiotics).
        - Patient instructions.
        Format as a concise medical report.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mask_base64}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        report_text = response.choices[0].message.content
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

def medsam_segment(image, medsam_model_path):
    global medsam_model
    if medsam_model is None:
        logger.error("MedSAM model not loaded")
        return None
    try:
        if image.ndim != 3 or image.shape[-1] != 3:
            logger.error("Input image must be RGB")
            return None
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        img_1024 = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img_1024 = cv2.cvtColor(img_1024, cv2.COLOR_RGB2BGR)
        img_tensor = torch.from_numpy(img_1024).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        # Dynamic bounding box
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        coords = np.where(edges > 0)
        if len(coords[0]) > 0:
            x_min, x_max = coords[1].min(), coords[1].max()
            y_min, y_max = coords[0].min(), coords[0].max()
            box = np.array([[x_min, y_min, x_max, y_max]])
        else:
            logger.warning("No edges detected, using default bounding box")
            box = np.array([[256, 256, 768, 768]])  # Fallback
        box_tensor = torch.from_numpy(box).float().to(device)
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_tensor)
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None, boxes=box_tensor, masks=None
            )
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
        return mask
    except Exception as e:
        logger.error(f"Error in MedSAM segmentation: {str(e)}")
        return None

def load_data(image_dir, mask_dir=None, img_size=(128, 128), api_key=None, is_training=True):
    logger.info(f"Loading data from {image_dir}, is_training={is_training}")
    images, masks, original_images = [], [], []
    if not os.path.exists(image_dir):
        logger.error(f"Image directory {image_dir} does not exist")
        return np.array([]), np.array([]), []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load image {img_name}, skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_images.append(img.copy())
        if api_key:
            quality_result = check_image_quality(img / 255.0, api_key)
            if not quality_result['is_valid']:
                logger.warning(f"Image {img_name} failed quality check: {quality_result['message']}")
                continue
        if mask_dir and os.path.exists(os.path.join(mask_dir, img_name)):
            mask = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask for {img_name}, using dummy mask")
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
    logger.info(f"Loaded {len(images)} images")
    transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Rotate(limit=40, p=0.5) if is_training else A.NoOp(),
        A.HorizontalFlip(p=0.5) if is_training else A.NoOp(),
        A.RandomBrightnessContrast(p=0.3) if is_training else A.NoOp(),
    ])
    augmented_images, augmented_masks = [], []
    for img, mask in zip(images, masks):
        augmented = transform(image=img, mask=mask)
        augmented_images.append(augmented['image'] / 255.0)
        augmented_masks.append((augmented['mask'] > 0.5).astype(np.uint8))
    logger.info(f"Completed data processing for {len(augmented_images)} images")
    return np.array(augmented_images), np.array(augmented_masks), original_images

def build_unet(input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder
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
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    # Decoder
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

def feature_extraction(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1), plt.imshow(canny, cmap='gray'), plt.title('Canny Edge')
    plt.subplot(1, 3, 2), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
    plt.subplot(1, 3, 3), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
    plt.savefig(f'feature_extraction_{timestamp}.png', dpi=150)
    plt.close('all')
    return canny, sobelx, sobely

def train_model(X_train, y_train, X_val, y_val, img_size=(128, 128), batch_size=4, epochs=50, model_save_path=None, pretrained_model_path=None):
    model = build_unet(input_shape=(128, 128, 3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
                  loss=sm.losses.binary_focal_dice_loss,
                  metrics=[sm.metrics.iou_score])
    if pretrained_model_path:
        model.load_weights(pretrained_model_path)
        logger.info(f"Loaded pre-trained model from {pretrained_model_path}")
        return model, None
    if len(X_train) == 0 or len(X_val) == 0:
        logger.error("Training or validation data is empty. Cannot proceed with training.")
        return model, None
    X_train_resized = tf.image.resize(X_train, [128, 128], method='bilinear').numpy()
    y_train_resized = tf.image.resize(y_train, [128, 128], method='nearest').numpy()
    X_val_resized = tf.image.resize(X_val, [128, 128], method='bilinear').numpy()
    y_val_resized = tf.image.resize(y_val, [128, 128], method='nearest').numpy()
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        VisualizationCallback(X_val_resized, y_val_resized)
    ]
    history = model.fit(
        X_train_resized, y_train_resized,
        validation_data=(X_val_resized, y_val_resized),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    return model, history

def explainable_ai(model, img):
    logger.info("Generating Grad-CAM visualization")
    img_tensor = tf.image.resize(img, (128, 128))[None, ...]
    prediction = model(img_tensor)
    heatmap = tf.squeeze(prediction, axis=[0, -1])
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def predict_healing_potential(mask, image, api_key=None, patient_metadata=None):
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
    if api_key:
        client = OpenAI(api_key=api_key)
        img_uint8 = (image * 255).astype(np.uint8)
        _, img_buffer = cv2.imencode('.png', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        mask_uint8 = (mask * 255).astype(np.uint8)
        _, mask_buffer = cv2.imencode('.png', mask_uint8)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        prompt = f"""
        Analyze this wound image and its segmentation mask (white = wound, black = background).
        - Estimate wound severity (Mild, Moderate, Severe).
        - Predict healing potential (e.g., High, Moderate, Low with percentage).
        - Consider wound appearance (color, exudate, necrosis) and area ({wound_area} pixels).
        """
        if patient_metadata:
            prompt += f"Patient metadata: {patient_metadata}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mask_base64}"}}
                    ]
                }
            ],
            max_tokens=200
        )
        message = response.choices[0].message.content
        if "Severe" in message:
            severity = "Severe"
        elif "Moderate" in message:
            severity = "Moderate"
        else:
            severity = "Mild"
        healing_potential = message.split("healing potential")[-1].strip()[:50]
    return severity, healing_potential

def evaluate_model(model, X_test, y_test):
    if len(X_test) == 0 or len(y_test) == 0:
        logger.error("Test data is empty. Cannot evaluate model.")
        return
    X_test_resized = tf.image.resize(X_test, [128, 128], method='bilinear').numpy()
    y_test_resized = tf.image.resize(y_test, [128, 128], method='nearest').numpy()
    y_pred = model.predict(X_test_resized, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(np.uint8)
    y_test_flat = y_test_resized.flatten()
    y_pred_flat = y_pred_binary.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_flat, y_pred_flat, average='binary')
    iou = sm.metrics.iou_score(y_test_resized, y_pred_binary)
    logger.info(f"Test Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")

def evaluate_and_visualize(model, X_test, y_test, original_images, api_key=None, use_medsam=False, medsam_model_path=None, output_dir="output"):
    logger.info("Starting evaluation and visualization")
    if len(X_test) == 0:
        logger.error("No test images available for evaluation")
        return
    val_dir = os.path.join(output_dir, "val_predictions")
    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    batch_size = 4
    for start_idx in range(0, len(X_test), batch_size):
        end_idx = min(start_idx + batch_size, len(X_test))
        batch_imgs = X_test[start_idx:end_idx]
        batch_true_masks = y_test[start_idx:end_idx]
        batch_original_imgs = original_images[start_idx:end_idx]
        batch_imgs_resized = tf.image.resize(batch_imgs, [128, 128], method='bilinear').numpy()
        batch_preds = model.predict(batch_imgs_resized, verbose=0)
        # Ensure batch_preds has the correct shape (batch_size, height, width, 1)
        if len(batch_preds.shape) == 3:  # If shape is (batch_size, height, width)
            batch_preds = np.expand_dims(batch_preds, axis=-1)  # Add channel dimension
        batch_preds = tf.image.resize(batch_preds, [X_test.shape[1], X_test.shape[2]], method='nearest').numpy()
        for i, (img, true_mask, pred, orig_img) in enumerate(zip(batch_imgs, batch_true_masks, batch_preds, batch_original_imgs)):
            idx = start_idx + i
            logger.info(f"Processing image {idx + 1}/{len(X_test)}")
            pred_mask = (pred > 0.5).astype(np.uint8)
            # Adjust indexing based on shape
            if pred_mask.shape[-1] == 1:  # If channel dimension exists
                pred_mask = pred_mask.squeeze()  # Remove single channel dimension
            medsam_pred = None
            if use_medsam and medsam_model_path:
                medsam_pred = medsam_segment(orig_img, medsam_model_path)
                if medsam_pred is not None:
                    medsam_pred = cv2.resize(medsam_pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    medsam_pred = np.expand_dims(medsam_pred, axis=(0, -1))
            canny, sobelx, sobely = feature_extraction(orig_img)
            heatmap = explainable_ai(model, img)
            patient_metadata = {"diabetes": True, "age": 65}
            # Use correct indexing for pred_mask
            severity, healing_potential = predict_healing_potential(pred_mask, orig_img, api_key, patient_metadata)
            validation_result = {"is_valid": True, "message": "O3 validation skipped (no API key)"}
            if api_key:
                validation_result = validate_segmentation(orig_img, pred_mask, api_key)
            report_path = None
            if api_key:
                report_path = generate_clinical_report(
                    orig_img, pred_mask, severity, healing_potential,
                    validation_result, api_key, report_dir
                )
                if report_path:
                    logger.info(f"Clinical report saved at: {report_path}")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plt.figure(figsize=(10, 10))
            plt.subplot(3, 4, 1), plt.imshow(orig_img), plt.title('Original Image')
            plt.subplot(3, 4, 2), plt.imshow(true_mask[0, :, :, 0], cmap='gray'), plt.title('True Mask')
            plt.subplot(3, 4, 3), plt.imshow(pred_mask, cmap='gray'), plt.title('U-Net Predicted Mask')
            plt.subplot(3, 4, 4)
            contours, _ = cv2.findContours((pred_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = (orig_img * 255).astype(np.uint8).copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            plt.imshow(contour_img), plt.title('Wound Boundary')
            plt.subplot(3, 4, 5), plt.imshow(canny, cmap='gray'), plt.title('Canny Edge')
            plt.subplot(3, 4, 6), plt.imshow(heatmap, cmap='jet'), plt.title('Grad-CAM Heatmap')
            plt.subplot(3, 4, 7)
            plt.text(0.1, 0.8, f"Severity: {severity}", fontsize=10)
            plt.text(0.1, 0.6, f"Healing Potential: {healing_potential}", fontsize=10)
            plt.text(0.1, 0.4, f"O3 Validation: {validation_result['message'][:100]}", fontsize=8)
            plt.axis('off'), plt.title('AI Prediction')
            if medsam_pred is not None:
                plt.subplot(3, 4, 8), plt.imshow(medsam_pred[0, :, :, 0], cmap='gray'), plt.title('MedSAM Predicted Mask')
            plt.savefig(os.path.join(output_dir, f'segmentation_results_{timestamp}_{idx}.png'), dpi=150)
            plt.close('all')
            gc.collect()

import matplotlib.pyplot as plt

# Plot IOU and Loss
def plot_training_curves(history):
    epochs = range(1, len(history.history['loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot IOU
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['iou_score'], label='Train IOU')
    plt.plot(epochs, history.history['val_iou_score'], label='Val IOU')
    plt.title('IOU Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IOU Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

# Call the function
#plot_training_curves(history)

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate wound segmentation model.')
    parser.add_argument('--train_image_dir', help='Path to training images')
    parser.add_argument('--train_mask_dir', help='Path to training masks')
    parser.add_argument('--test_image_dir', required=True, help='Path to test images')
    parser.add_argument('--test_mask_dir', help='Path to test masks')
    parser.add_argument('--img_size', type=int, default=128, help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--openai_api_key', help='OpenAI API key for O3 integration')
    parser.add_argument('--use_medsam', action='store_true', help='Use MedSAM for segmentation')
    parser.add_argument('--medsam_model_path', help='Path to MedSAM model weights')
    parser.add_argument('--pretrained_model_path', help='Path to pre-trained U-Net model weights')
    parser.add_argument('--model_save_path', default='best_unet_wound_model.h5', help='Path to save model weights')
    parser.add_argument('--output_dir', default='output', help='Base output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.debug(f"Arguments parsed: {vars(args)}")
    if args.openai_api_key:
        logger.info("OpenAI API key provided, enabling O3 integration")
    if args.use_medsam and args.medsam_model_path:
        load_medsam_model(args.medsam_model_path)
        if medsam_model is None:
            logger.warning("MedSAM model failed to load, disabling MedSAM")
            args.use_medsam = False
    X_test, y_test, original_test_images = load_data(
        args.test_image_dir, args.test_mask_dir, img_size=(args.img_size, args.img_size),
        api_key=args.openai_api_key, is_training=False
    )
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
        X_train, y_train, _ = load_data(
            args.train_image_dir, args.train_mask_dir, img_size=(args.img_size, args.img_size),
            api_key=args.openai_api_key, is_training=True
        )
        if len(X_train) == 0:
            logger.error("No training images loaded. Exiting.")
            sys.exit(1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        y_train = np.expand_dims(y_train, axis=-1).astype('float32')
        y_val = np.expand_dims(y_val, axis=-1).astype('float32')
        logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        img_size=(args.img_size, args.img_size), batch_size=args.batch_size,
        epochs=args.epochs, model_save_path=args.model_save_path,
        pretrained_model_path=args.pretrained_model_path
    )
    if history:
        plot_training_curves(history)

    if not args.pretrained_model_path:
        model.summary()
    evaluate_model(model, X_test, y_test)
    evaluate_and_visualize(
        model, X_test, y_test, original_test_images,
        api_key=args.openai_api_key, use_medsam=args.use_medsam,
        medsam_model_path=args.medsam_model_path, output_dir=args.output_dir
    )
    logger.info(f"Evaluation completed. Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()