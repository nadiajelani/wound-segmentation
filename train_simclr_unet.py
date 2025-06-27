import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import logging

# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIG ===
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 50
DATASET_PATH = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images"
MODEL_SAVE_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_wound_segmentation.keras"

# === CUSTOM LOSS ===
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# === LOAD DATA ===
def load_images_and_masks(image_dir, mask_dir, target_size=(224, 224), mask_suffix=None):
    images = []
    masks = []
    image_filenames = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"Loading from {image_dir}: {len(image_filenames)} files found")
    corrupted_files = []
    missing_mask_files = []

    for filename in image_filenames:
        img_path = os.path.join(image_dir, filename)
        mask_filename = filename if mask_suffix is None else filename.replace(os.path.splitext(filename)[1], mask_suffix)
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            logger.warning(f"⚠️ Mask not found for {filename} at {mask_path}, skipping.")
            missing_mask_files.append(filename)
            continue

        try:
            # Verify image file
            with Image.open(img_path) as img_pil:
                img_pil.verify()
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img) / 255.0

            # Verify mask file
            with Image.open(mask_path) as mask_pil:
                mask_pil.verify()
            mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
            mask = img_to_array(mask)
            mask = (mask > 127).astype(np.float32)

            images.append(img)
            masks.append(mask)
            if len(image_filenames) <= 1000:
                logger.info(f"✅ Loaded: {filename} with mask {mask_filename}")
        except (OSError, UnidentifiedImageError) as e:
            logger.error(f"❌ Failed to load {filename}: {e}")
            corrupted_files.append(filename)
            continue

    if corrupted_files:
        logger.warning(f"⚠️ Corrupted files detected: {corrupted_files}")
    if missing_mask_files:
        logger.warning(f"⚠️ Missing mask files: {missing_mask_files}")
    logger.info(f"Loaded {len(images)} image-mask pairs from {image_dir}")
    return np.array(images), np.array(masks)

# === LOAD DATASET ===
image_dir = os.path.join(DATASET_PATH, "train_images")
mask_dir = os.path.join(DATASET_PATH, "train_masks")
test_image_dir = os.path.join(DATASET_PATH, "test_images")
test_mask_dir = os.path.join(DATASET_PATH, "test_masks")

# Verify directories exist
for dir_path in [image_dir, mask_dir, test_image_dir, test_mask_dir]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

X_train_full, y_train_full = load_images_and_masks(image_dir, mask_dir)
if len(X_train_full) == 0:
    raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
X_test, y_test = load_images_and_masks(test_image_dir, test_mask_dir)
if len(X_test) == 0:
    logger.warning(f"⚠️ No valid test data found in {test_image_dir} and {test_mask_dir}")

logger.info(f"Loaded {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples.")

# === DATA AUGMENTATION ===
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    seed=42
)

# === LOAD MODEL ===
logger.info("Loading model...")
try:
    model = load_model(MODEL_SAVE_PATH, custom_objects={'combined_loss': combined_loss})
    logger.info(f"✅ Loaded model from {MODEL_SAVE_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]  # Reverted to MeanIoU
)

# === RESUME TRAINING ===
logger.info("Resuming training from epoch 14...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)
history = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    initial_epoch=39,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# === EVALUATE ON TEST SET ===
if len(X_test) > 0:
    logger.info("Evaluating on test set...")
    test_loss, test_acc, test_iou = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    logger.info(f"Test Accuracy: {test_acc:.4f}, Test IoU: {test_iou:.4f}")
else:
    logger.warning(f"⚠️ Skipping test set evaluation due to empty test set.")

# === SAVE ===
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
logger.info(f"✅ Model saved to {MODEL_SAVE_PATH}")

# === VISUALIZE PREDICTIONS ===
import matplotlib.pyplot as plt

def visualize_predictions(model, X_test, y_test, num_samples=3):
    predictions = model.predict(X_test[:num_samples])
    predictions = (predictions > 0.5).astype(np.float32)
    for i in range(num_samples):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i])
        plt.title("Input Image")
        plt.subplot(1, 3, 2)
        plt.imshow(y_test[i, :, :, 0], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.title("Predicted Mask")
        plt.show()

if len(X_test) > 0:
    logger.info("Visualizing predictions...")
    visualize_predictions(model, X_test, y_test)