#!/usr/bin/env python3
"""
SimCLR U-Net with 5-Fold Cross-Validation

This script adds cross-validation to your existing SimCLR U-Net training.
"""

import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, concatenate, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
import json

# Import your existing functions
from train_simclr_unet import (
    load_images_and_masks, crop_wound_patch, PatchGenerator, 
    build_unet, total_loss, wound_dice_coef
)

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# CONFIGURATION
IMG_CHANNELS = 3
CROP_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 15
DATASET_PATH = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images"
MODEL_SAVE_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_cv"

class FoldMetricsCallback(tf.keras.callbacks.Callback):
    """Callback to capture fold-wise metrics."""
    
    def __init__(self, fold_number: int):
        self.fold_number = fold_number
        self.metrics_history = {
            'fold': fold_number,
            'val_dice': [],
            'val_iou': [],
            'val_loss': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for key in ['val_dice_coef', 'val_loss']:
                if key in logs:
                    self.metrics_history[key].append(float(logs[key]))
    
    def on_train_end(self, logs=None):
        # Save fold metrics
        filename = f"fold_{self.fold_number}_metrics.json"
        with open(filename, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"✅ Fold {self.fold_number} metrics saved")

def main():
    print("🔄 Loading data for cross-validation...")
    
    # Load data (same as your original script)
    image_dir = os.path.join(DATASET_PATH, "train_images")
    mask_dir = os.path.join(DATASET_PATH, "train_masks")
    X_full, y_full = load_images_and_masks(image_dir, mask_dir)
    
    # Crop wound-centered patches
    X_crops, y_crops = [], []
    for img, msk in zip(X_full, y_full):
        img_patch, msk_patch = crop_wound_patch(img, msk, crop_size=CROP_SIZE)
        X_crops.append(img_patch)
        y_crops.append(msk_patch)
    X_crops, y_crops = np.stack(X_crops), np.stack(y_crops)
    y_crops = y_crops[:, :, :, :1]
    
    print(f"✅ Loaded {len(X_crops)} wound patches")
    
    # 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_crops)):
        print(f"\n🔄 Training Fold {fold + 1}/5...")
        
        X_train_fold, X_val_fold = X_crops[train_idx], X_crops[val_idx]
        y_train_fold, y_val_fold = y_crops[train_idx], y_crops[val_idx]
        
        # Create data generators
        train_gen = PatchGenerator(X_train_fold, y_train_fold, BATCH_SIZE, augment=True)
        val_gen = PatchGenerator(X_val_fold, y_val_fold, BATCH_SIZE, augment=False)
        
        # Build model
        model = build_unet(input_shape=(CROP_SIZE, CROP_SIZE, IMG_CHANNELS), num_classes=1, weights='imagenet')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=total_loss,
            metrics=[wound_dice_coef]
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint(f"{MODEL_SAVE_PATH}_fold{fold}.keras", monitor='val_loss', save_best_only=True),
            FoldMetricsCallback(fold_number=fold)
        ]
        
        # Train
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on validation set
        val_preds = model.predict(val_gen, verbose=0)
        val_preds_bin = (val_preds > 0.3).astype(np.float32)
        
        # Calculate metrics
        dice_scores = []
        iou_scores = []
        
        for i in range(len(val_preds_bin)):
            true_mask = y_val_fold[i, :, :, 0]
            pred_mask = val_preds_bin[i, :, :, 0]
            
            # Dice score
            intersection = np.sum(true_mask * pred_mask)
            dice = (2. * intersection) / (np.sum(true_mask) + np.sum(pred_mask) + 1e-6)
            dice_scores.append(dice)
            
            # IoU score
            union = np.sum(true_mask) + np.sum(pred_mask) - intersection
            iou = intersection / (union + 1e-6)
            iou_scores.append(iou)
        
        fold_mean_dice = np.mean(dice_scores)
        fold_mean_iou = np.mean(iou_scores)
        
        fold_results.append({
            'fold': fold,
            'dice': fold_mean_dice,
            'iou': fold_mean_iou,
            'n_samples': len(val_preds_bin)
        })
        
        print(f"✅ Fold {fold + 1} - Dice: {fold_mean_dice:.3f}, IoU: {fold_mean_iou:.3f}")
    
    # Save cross-validation results
    with open('simclr_unet_cv_results.json', 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    print("\n✅ Cross-validation complete!")
    print("📊 Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold'] + 1}: Dice={result['dice']:.3f}, IoU={result['iou']:.3f}")
    
    # Calculate overall statistics
    dice_values = [r['dice'] for r in fold_results]
    iou_values = [r['iou'] for r in fold_results]
    
    print(f"\n📈 Overall Results:")
    print(f"  Dice: {np.mean(dice_values):.3f} ± {np.std(dice_values):.3f}")
    print(f"  IoU:  {np.mean(iou_values):.3f} ± {np.std(iou_values):.3f}")

if __name__ == "__main__":
    main()
