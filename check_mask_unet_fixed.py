#!/usr/bin/env python3
"""
Fixed SimCLR U-Net Validation Script

This script properly loads your SimCLR U-Net model with all custom objects
and generates validation results for confidence interval calculation.
"""

import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score

# Import custom objects from your training script
@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Calculate Dice loss for segmentation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def binary_focal_loss(y_true, y_pred, gamma=1.0, alpha=0.1):
    """Calculate binary focal loss."""
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    focal_loss = alpha * tf.math.pow(1 - y_pred, gamma) * bce
    return tf.reduce_mean(focal_loss)

@tf.keras.utils.register_keras_serializable()
def total_loss(y_true, y_pred):
    """Combine binary cross-entropy, Dice loss, and focal loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    focal = binary_focal_loss(y_true, y_pred, gamma=1.0, alpha=0.1)
    return bce + dice + focal

@tf.keras.utils.register_keras_serializable()
def wound_dice_coef(y_true, y_pred, smooth=1):
    """Calculate wound Dice coefficient."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# CONFIGURATION
IMG_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_images/"
MASK_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_masks/"
MODEL_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
IMG_SIZE = (128, 128)  # Your model expects 128x128 input
THRESHOLD = 0.3  # Adaptive threshold from your training script

def dice_score(y_true, y_pred):
    """Calculate Dice score for validation."""
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics for a single prediction."""
    # Flatten for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    dice = dice_score(y_true, y_pred)
    iou = jaccard_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
    precision = precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    
    return dice, iou, precision, recall, accuracy

print("🔄 Loading SimCLR U-Net model...")

# Load model with custom objects
custom_objects = {
    "dice_coefficient": wound_dice_coef,
    "total_loss": total_loss,
    "dice_loss": dice_loss,
    "binary_focal_loss": binary_focal_loss
}

try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Check if directories exist
if not os.path.exists(IMG_DIR):
    print(f"❌ Image directory not found: {IMG_DIR}")
    exit(1)

if not os.path.exists(MASK_DIR):
    print(f"❌ Mask directory not found: {MASK_DIR}")
    exit(1)

print("🔄 Running validation...")

results = []
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])

if not image_files:
    print(f"❌ No images found in {IMG_DIR}")
    exit(1)

print(f"📊 Found {len(image_files)} test images")

for i, img_file in enumerate(image_files):
    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, img_file)

    if not os.path.exists(mask_path):
        print(f"⚠️ Skipping {img_file}: mask not found")
        continue

    try:
        # Load and preprocess image
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        # Load and preprocess true mask
        mask_true = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_true = cv2.resize(mask_true, IMG_SIZE)
        mask_true_bin = (mask_true > 127).astype(np.uint8)

        # Predict mask
        pred_mask = model.predict(img_input, verbose=0)[0, :, :, 0]
        
        # Apply adaptive thresholding (from your training script)
        mean_pred = np.mean(pred_mask)
        adaptive_threshold = max(0.1, min(0.9, THRESHOLD + (mean_pred - 0.5) * 0.2))
        pred_mask_bin = (pred_mask > adaptive_threshold).astype(np.uint8)

        # Skip if both masks are empty (IoU can't be computed)
        if np.sum(mask_true_bin) == 0 and np.sum(pred_mask_bin) == 0:
            dice = 1.0
            iou = 1.0
            precision = 1.0
            recall = 1.0
            accuracy = 1.0
        else:
            dice, iou, precision, recall, accuracy = calculate_metrics(mask_true_bin, pred_mask_bin)

        results.append({
            "Image": img_file,
            "Dice Score": round(dice, 4),
            "IoU Score": round(iou, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Accuracy": round(accuracy, 4),
            "Threshold": round(adaptive_threshold, 3)
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images...")

    except Exception as e:
        print(f"⚠️ Skipping {img_file} due to error: {e}")
        results.append({
            "Image": img_file,
            "Dice Score": "Error",
            "IoU Score": "Error",
            "Precision": "Error",
            "Recall": "Error",
            "Accuracy": "Error",
            "Threshold": "Error"
        })

# Save results
df = pd.DataFrame(results)
output_file = "simclr_unet_real_validation_results.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Validation complete! Results saved to '{output_file}'")
print(f"📊 Processed {len(results)} images")

# Display summary statistics
valid_results = df[df['Dice Score'] != 'Error']
if len(valid_results) > 0:
    print(f"\n📈 Summary Statistics ({len(valid_results)} valid samples):")
    print("=" * 50)
    
    for metric in ['Dice Score', 'IoU Score', 'Precision', 'Recall', 'Accuracy']:
        values = pd.to_numeric(valid_results[metric], errors='coerce')
        mean_val = values.mean()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        
        print(f"{metric}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  Min:  {min_val:.4f}")
        print(f"  Max:  {max_val:.4f}")
        print()

print("🎯 Ready for confidence interval calculation!")
print("Run: python simclr_unet_confidence_intervals.py")

