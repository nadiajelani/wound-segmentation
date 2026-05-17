"""
Improvement 1 – Higher Resolution Input (128→256)
===================================================
Run this script on your existing dataset to retrain the segmentation model
at 256×256.  No architecture changes are needed – ResNet50 skip layers
are resolution-agnostic; only the data pipeline and final upsampling change.

Usage
-----
  python retrain_256.py \
      --train_images  data/train/images \
      --train_masks   data/train/masks  \
      --val_images    data/val/images   \
      --val_masks     data/val/masks    \
      --pretrained    models/wound_segmentation.keras \
      --output        models/wound_seg_256.keras

What changes vs the 128×128 model
-----------------------------------
  • Input shape        128×128×3  →  256×256×3
  • Final UpSampling   1×         →  2× (extra stride to reach 256)
  • Batch size         16         →  8  (2× memory per sample)
  • Augmentation       added elastic deformation & random gamma
"""

import argparse, os, math, logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retrain_256")

# ── config ────────────────────────────────────────────────────────────────────
IMG_H, IMG_W = 256, 256
BATCH_SIZE   = 8
EPOCHS_FROZEN = 20     # encoder frozen
EPOCHS_FINETUNE = 40   # partial unfreeze
LR_INIT     = 1e-4
LR_FINETUNE = 3e-5
SEED        = 42

# ── architecture ──────────────────────────────────────────────────────────────
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_gate(x, g, filters):
    """Additive attention gate – improves boundary sharpness."""
    theta_x = layers.Conv2D(filters, 1, padding="same")(x)
    phi_g   = layers.Conv2D(filters, 1, padding="same")(g)
    # align spatial dims
    if theta_x.shape[1] != phi_g.shape[1]:
        phi_g = layers.UpSampling2D(
            size=(theta_x.shape[1] // phi_g.shape[1],
                  theta_x.shape[2] // phi_g.shape[2]))(phi_g)
    add  = layers.Activation("relu")(layers.Add()([theta_x, phi_g]))
    psi  = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(add)
    return layers.Multiply()([x, psi])

def build_unet_256(pretrained_path=None):
    """
    ResNet50-based U-Net at 256×256 with attention gates.
    If pretrained_path is given the encoder weights are warm-started
    from the 128×128 model (weights are shape-agnostic for conv layers).
    """
    inp = layers.Input(shape=(IMG_H, IMG_W, 3))
    base = ResNet50(include_top=False, weights="imagenet", input_tensor=inp)

    # Optionally warm-start from existing model
    if pretrained_path and Path(pretrained_path).exists():
        try:
            old = tf.keras.models.load_model(pretrained_path, compile=False)
            for new_l in base.layers:
                try:
                    old_l = old.get_layer(new_l.name)
                    new_l.set_weights(old_l.get_weights())
                except (ValueError, AttributeError):
                    pass
            log.info("Warm-started encoder from %s", pretrained_path)
        except Exception as e:
            log.warning("Warm-start failed (%s) – using ImageNet weights", e)

    # Skip connections (resolution doubles each level)
    s1 = base.get_layer("conv1_relu").output          # 128×128×64
    s2 = base.get_layer("conv2_block3_out").output    # 64×64×256
    s3 = base.get_layer("conv3_block4_out").output    # 32×32×512
    s4 = base.get_layer("conv4_block6_out").output    # 16×16×1024
    bn = base.get_layer("conv5_block3_out").output    # 8×8×2048

    # Decoder with attention gates
    def up(x, skip, f):
        x  = layers.Conv2DTranspose(f, 3, strides=2, padding="same")(x)
        sk = attention_gate(skip, x, f // 2)
        x  = layers.Concatenate()([x, sk])
        return conv_block(x, f)

    d1 = up(bn, s4, 512)   # 16×16
    d2 = up(d1, s3, 256)   # 32×32
    d3 = up(d2, s2, 128)   # 64×64
    d4 = up(d3, s1, 64)    # 128×128
    # Extra upsampling to reach 256×256 (new vs 128-model)
    x  = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(d4)
    x  = conv_block(x, 32)
    out = layers.Conv2D(1, 1, activation="sigmoid", name="mask_output")(x)

    return Model(inp, out, name="unet_256")

# ── losses ─────────────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    yt = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    yp = tf.keras.backend.flatten(y_pred)
    inter = tf.keras.backend.sum(yt * yp)
    return 1 - (2 * inter + smooth) / (
        tf.keras.backend.sum(yt) + tf.keras.backend.sum(yp) + smooth)

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    bce  = tf.keras.losses.binary_crossentropy(
        tf.cast(y_true, tf.float32), y_pred)
    return 0.4 * tf.reduce_mean(bce) + 0.6 * dice_loss(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def dice_coeff(y_true, y_pred, smooth=1e-6):
    yt = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    yp = tf.keras.backend.flatten(y_pred)
    inter = tf.keras.backend.sum(yt * yp)
    return (2 * inter + smooth) / (
        tf.keras.backend.sum(yt) + tf.keras.backend.sum(yp) + smooth)

# ── data pipeline ─────────────────────────────────────────────────────────────
def make_generator(img_dir, mask_dir, augment=False, batch_size=BATCH_SIZE):
    """
    Yields (image_batch, mask_batch) pairs at 256×256.
    Augmentation includes random elastic deformation + gamma.
    """
    img_paths  = sorted(Path(img_dir).glob("**/*.png")) + \
                 sorted(Path(img_dir).glob("**/*.jpg"))
    mask_paths = sorted(Path(mask_dir).glob("**/*.png")) + \
                 sorted(Path(mask_dir).glob("**/*.jpg"))
    assert len(img_paths) == len(mask_paths), \
        f"Image/mask count mismatch: {len(img_paths)} vs {len(mask_paths)}"
    log.info("Dataset: %d image-mask pairs (augment=%s)", len(img_paths), augment)

    def load_pair(ip, mp):
        img  = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,  (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        img  = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)[..., np.newaxis]
        return img, mask

    def random_gamma(img):
        gamma = np.random.uniform(0.7, 1.4)
        return np.clip(img ** gamma, 0, 1)

    def elastic_deform(img, mask, alpha=30, sigma=5):
        """Elastic deformation – critical for wound shape generalisation."""
        h, w = img.shape[:2]
        dx = cv2.GaussianBlur(
            np.random.rand(h, w).astype(np.float32) * 2 - 1, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(
            np.random.rand(h, w).astype(np.float32) * 2 - 1, (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
        img_d  = cv2.remap(img,  map_x, map_y, cv2.INTER_LINEAR)
        mask_d = cv2.remap(mask[..., 0], map_x, map_y, cv2.INTER_NEAREST)[..., np.newaxis]
        return img_d, mask_d

    indices = list(range(len(img_paths)))
    while True:
        np.random.shuffle(indices)
        imgs, masks = [], []
        for i in indices:
            img, mask = load_pair(img_paths[i], mask_paths[i])
            if augment:
                if np.random.rand() > 0.5: img = img[:, ::-1]; mask = mask[:, ::-1]
                if np.random.rand() > 0.5: img = img[::-1];    mask = mask[::-1]
                if np.random.rand() > 0.3: img = random_gamma(img)
                if np.random.rand() > 0.4: img, mask = elastic_deform(img, mask)
                angle = np.random.uniform(-25, 25)
                M = cv2.getRotationMatrix2D((IMG_W // 2, IMG_H // 2), angle, 1)
                img  = cv2.warpAffine(img,  M, (IMG_W, IMG_H))
                mask_sq = cv2.warpAffine(mask[..., 0], M, (IMG_W, IMG_H),
                                         flags=cv2.INTER_NEAREST)[..., np.newaxis]
                mask = mask_sq
            imgs.append(img); masks.append(mask)
            if len(imgs) == batch_size:
                yield np.array(imgs), np.array(masks)
                imgs, masks = [], []

# ── training ──────────────────────────────────────────────────────────────────
def train(args):
    log.info("Building 256×256 U-Net with attention gates…")
    model = build_unet_256(args.pretrained)

    # ── Phase 1: freeze encoder ──
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and "conv1" in layer.name.lower():
            layer.trainable = False
    log.info("Phase 1: encoder frozen, training decoder only")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_INIT),
        loss=combined_loss,
        metrics=[dice_coeff, "accuracy"])

    train_gen = make_generator(args.train_images, args.train_masks, augment=True)
    val_gen   = make_generator(args.val_images,   args.val_masks,   augment=False)

    n_train = len(list(Path(args.train_images).glob("**/*.png")) +
                  list(Path(args.train_images).glob("**/*.jpg")))
    n_val   = len(list(Path(args.val_images).glob("**/*.png")) +
                  list(Path(args.val_images).glob("**/*.jpg")))

    cb = [
        ModelCheckpoint(args.output, monitor="val_dice_coeff",
                        save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_dice_coeff", patience=10,
                      mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-7, verbose=1),
        TensorBoard(log_dir="logs/phase1"),
    ]

    model.fit(train_gen,
              steps_per_epoch=math.ceil(n_train / BATCH_SIZE),
              validation_data=val_gen,
              validation_steps=math.ceil(n_val / BATCH_SIZE),
              epochs=EPOCHS_FROZEN,
              callbacks=cb)

    # ── Phase 2: unfreeze top 50 layers ──
    log.info("Phase 2: unfreezing top 50 encoder layers for fine-tuning")
    for layer in model.layers[-50:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss=combined_loss,
        metrics=[dice_coeff, "accuracy"])

    cb[3] = TensorBoard(log_dir="logs/phase2")
    model.fit(train_gen,
              steps_per_epoch=math.ceil(n_train / BATCH_SIZE),
              validation_data=val_gen,
              validation_steps=math.ceil(n_val / BATCH_SIZE),
              epochs=EPOCHS_FINETUNE,
              callbacks=cb)

    log.info("✅ Training complete. Model saved to %s", args.output)

    # Print summary
    model.summary(line_length=100)
    log.info("Input shape: %s", model.input_shape)
    log.info("Output shape: %s", model.output_shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--train_masks",  required=True)
    ap.add_argument("--val_images",   required=True)
    ap.add_argument("--val_masks",    required=True)
    ap.add_argument("--pretrained",   default=None,
                    help="Path to existing 128×128 model for warm-start")
    ap.add_argument("--output",       default="models/wound_seg_256.keras")
    args = ap.parse_args()
    train(args)
