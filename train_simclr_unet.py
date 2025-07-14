
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
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- CONFIG ---
IMG_CHANNELS = 3
CROP_SIZE = 128    # Adjustable: 128, 160, 96
BATCH_SIZE = 4     # Increased for efficiency
EPOCHS = 15        # Reduced to fit 30-minute goal
DATASET_PATH = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images"
MODEL_SAVE_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"

# --- DATA LOADING HELPERS ---
def load_images_and_masks(image_dir, mask_dir, target_size=(224, 224)):
    images, masks = [], []
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for filename in files:
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path): continue
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img) / 255.0
        mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
        mask = img_to_array(mask)
        mask = (mask > 127).astype(np.float32)  # binarize
        images.append(img)
        masks.append(mask)
    return np.array(images), np.array(masks)

# --- CROP PATCHES AROUND WOUND ---
def crop_wound_patch(image, mask, crop_size=128, context=16):
    h, w = mask.shape[:2]
    mask_bin = (mask[..., 0] > 0.5).astype(np.uint8)
    coords = np.column_stack(np.where(mask_bin))
    if coords.shape[0] > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
    else:  # No wound: random center with safety buffer
        cy, cx = np.random.randint(context, h-context), np.random.randint(context, w-context)
    half = crop_size // 2
    y1, y2 = max(0, cy-half-context), min(h, cy+half+context)
    x1, x2 = max(0, cx-half-context), min(w, cx+half+context)
    img_patch = image[y1:y2, x1:x2, :]
    mask_patch = mask[y1:y2, x1:x2, :]
    img_patch = tf.image.resize_with_pad(img_patch, crop_size, crop_size)
    mask_patch = tf.image.resize_with_pad(mask_patch, crop_size, crop_size)
    return img_patch.numpy(), mask_patch.numpy()

# --- LOAD & PROCESS DATA ---
print("Loading data...")
image_dir = os.path.join(DATASET_PATH, "train_images")
mask_dir = os.path.join(DATASET_PATH, "train_masks")
X_full, y_full = load_images_and_masks(image_dir, mask_dir)

# --- Crop wound-centered patches ---
X_crops, y_crops = [], []
for img, msk in zip(X_full, y_full):
    img_patch, msk_patch = crop_wound_patch(img, msk, crop_size=CROP_SIZE)
    X_crops.append(img_patch)
    y_crops.append(msk_patch)
X_crops, y_crops = np.stack(X_crops), np.stack(y_crops)
y_crops = y_crops[:, :, :, :1]  # Ensure single channel

# --- VISUALIZE RANDOM PATCHES ---
print("Visualizing random crops (ensure wounds are visible!)")
idxs = np.random.choice(len(X_crops), min(12, len(X_crops)), replace=False)
for idx in idxs:
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.imshow(X_crops[idx]); plt.title("Patch")
    plt.subplot(1, 2, 2); plt.imshow(y_crops[idx, :, :, 0], cmap='gray'); plt.title("Mask")
    plt.show()

# --- SPLIT ---
X_train, X_val, y_train, y_val = train_test_split(X_crops, y_crops, test_size=0.2, random_state=42)

# --- DATA AUGMENTATION ---
def augment_batch(Xb, yb):
    Xb_aug, yb_aug = [], []
    for x, y in zip(Xb, yb):
        if random.random() > 0.5:
            x = np.fliplr(x); y = np.fliplr(y)
        if random.random() > 0.5:
            x = np.flipud(x); y = np.flipud(y)
        Xb_aug.append(x)
        yb_aug.append(y)
    return np.array(Xb_aug), np.array(yb_aug)

class PatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=4, augment=True):
        self.X, self.y, self.bs, self.augment = X, y, batch_size, augment
        self.indices = np.arange(len(X))
    def __len__(self): return int(np.ceil(len(self.X) / self.bs))
    def __getitem__(self, idx):
        inds = self.indices[idx * self.bs:(idx + 1) * self.bs]
        Xb, yb = self.X[inds], self.y[inds]
        if self.augment:
            Xb, yb = augment_batch(Xb, yb)
        return Xb, yb

train_gen = PatchGenerator(X_train, y_train, batch_size=BATCH_SIZE, augment=True)
val_gen = PatchGenerator(X_val, y_val, batch_size=BATCH_SIZE, augment=False)

# --- UNET DEFINITION ---
def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def decoder_block(input_tensor, skip_tensor, num_filters):
    x = UpSampling2D((2, 2))(input_tensor)
    x = concatenate([x, skip_tensor])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(128, 128, 3), num_classes=1, weights='imagenet'):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights=weights, input_tensor=inputs)
    skip1 = base_model.get_layer("conv1_relu").output        # 64x64
    skip2 = base_model.get_layer("conv2_block3_out").output  # 32x32
    skip3 = base_model.get_layer("conv3_block4_out").output  # 16x16
    skip4 = base_model.get_layer("conv4_block6_out").output  # 8x8
    bottleneck = base_model.get_layer("conv5_block3_out").output  # 4x4

    d1 = decoder_block(bottleneck, skip4, 512)
    d2 = decoder_block(d1, skip3, 256)
    d3 = decoder_block(d2, skip2, 128)
    d4 = decoder_block(d3, skip1, 64)
    x = UpSampling2D((2, 2))(d4)  # 64 -> 128
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# --- LOSS AND METRIC ---
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
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
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

# --- BUILD & TRAIN ---
print("Building model...")
model = build_unet(input_shape=(CROP_SIZE, CROP_SIZE, IMG_CHANNELS), num_classes=1, weights='imagenet')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=total_loss,
    metrics=[wound_dice_coef]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)

print("Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# --- VISUALIZE PREDICTIONS ---
def visualize_predictions(model, X_val, y_val, num_samples=6, threshold=0.3):
    """Visualize input patches, ground truth, and predicted masks with adaptive thresholding."""
    preds = model.predict(X_val[:num_samples])
    for i in range(num_samples):
        mean_pred = np.mean(preds[i])
        adaptive_threshold = max(0.1, min(0.9, threshold + (mean_pred - 0.5) * 0.2))
        preds_bin = (preds[i] > adaptive_threshold).astype(np.float32)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(X_val[i])
        plt.title("Input Patch")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(y_val[i, :, :, 0], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(preds_bin[:, :, 0], cmap='gray')
        plt.title(f"Predicted Mask (Threshold: {adaptive_threshold:.3f})")
        plt.axis('off')
        plt.show()

print("Visualizing predictions...")
visualize_predictions(model, X_val, y_val, num_samples=6)

print("✅ Training & evaluation complete. Model saved at:", MODEL_SAVE_PATH)
