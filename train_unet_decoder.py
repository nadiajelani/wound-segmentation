# train_simclr_unet.py

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from unet_decoder import build_simclr_unet

# === CONFIG ===
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMAGE_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/train_images"
MASK_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/train_masks"

# === Load and preprocess data ===
def load_images_and_masks(image_dir, mask_dir, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    images = []
    masks = []

    for file in image_files:
        img_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file)
        if not os.path.exists(mask_path):
            continue

        img = load_img(img_path, target_size=target_size)
        mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')

        img = img_to_array(img) / 255.0
        mask = img_to_array(mask) / 255.0
        mask = np.where(mask > 0.5, 1, 0)  # Binarize

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

print("Loading data...")
X, y = load_images_and_masks(IMAGE_DIR, MASK_DIR)
print(f"Loaded {len(X)} image-mask pairs")

# === Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Build Model ===
model = build_simclr_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Train ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16
)

# === Save Model ===
model.save("simclr_unet_wound_segmentation.keras")
print("Model saved as 'simclr_unet_wound_segmentation.keras'")
