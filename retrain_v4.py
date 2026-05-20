"""
quick_retrain.py
----------------
Fixes the stuck/slow training by pre-loading ALL images into RAM first,
then training from memory arrays. Much faster on M3 with OneDrive paths.

Run:
    cd /Users/nadiajelani/projects/wound-segmentation
    python quick_retrain.py
"""

import os, random, json, logging, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger()

# ── paths ─────────────────────────────────────────────────────────────────────
HOME = Path.home()
UNE  = HOME / "Library/CloudStorage/OneDrive-UniversityofNewEngland/Attachments/Wound-data/data"

STAGE_DIRS = [UNE / f"ulcers_self/Stage_{i}" for i in range(1, 5)]
DFU_DIR    = UNE / "DFU"
ISIC_DIR   = UNE / "ISIC-images"
PRETRAINED = Path("models/simclr_unet_patch_wound.keras")
OUT_SEG    = Path("models/wound_seg_256.keras")
OUT_STAGE  = Path("models/wound_stage_clf.keras")
OUT_TYPE   = Path("models/wound_type_clf.keras")

SZ   = 256
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── helpers ───────────────────────────────────────────────────────────────────
def find_images(folder, limit=None):
    folder = Path(folder)
    if not folder.exists():
        return []
    imgs = [p for p in folder.rglob("*")
            if p.suffix.lower() in EXTS
            and "venv" not in str(p)
            and "wound_seg_env" not in str(p)]
    if limit:
        random.shuffle(imgs)
        imgs = imgs[:limit]
    return imgs

def load_img(path):
    try:
        img = Image.open(str(path)).convert("RGB")
        img = img.resize((SZ, SZ), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except:
        return None

def make_mask(img):
    """Colour-based pseudo wound mask."""
    u8  = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    m1  = cv2.inRange(hsv, (0,   40, 60), (20,  255, 255))
    m2  = cv2.inRange(hsv, (155, 40, 60), (180, 255, 255))
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    m3  = (lab[:,:,1].astype(np.float32) > 135).astype(np.uint8) * 255
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    pct  = mask.sum() / (mask.shape[0] * mask.shape[1] * 255 + 1e-6)
    if pct < 0.01 or pct > 0.65:
        mask = np.zeros_like(mask)
        cv2.ellipse(mask, (SZ//2, SZ//2), (SZ//4, SZ//4), 0, 0, 360, 255, -1)
    return (mask > 0).astype(np.float32)[..., np.newaxis]

def augment(img, mask=None):
    if random.random() > 0.5:
        img = img[:, ::-1]; mask = mask[:, ::-1] if mask is not None else mask
    if random.random() > 0.5:
        img = img[::-1];    mask = mask[::-1]    if mask is not None else mask
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((SZ//2, SZ//2), angle, 1)
    img = cv2.warpAffine(img, M, (SZ, SZ))
    if mask is not None:
        mf = cv2.warpAffine(mask[..., 0], M, (SZ, SZ),
                             flags=cv2.INTER_NEAREST)[..., np.newaxis]
        mask = mf
    if random.random() > 0.4:
        img = np.clip(img ** random.uniform(0.7, 1.4), 0, 1)
    return img, mask

# ── pre-load into RAM ─────────────────────────────────────────────────────────
def preload_seg(max_total=3000):
    """Load images + pseudo masks into numpy arrays."""
    log.info(f"Pre-loading segmentation data (max {max_total})…")
    paths = []
    for d in STAGE_DIRS:
        paths.extend(find_images(d))
    paths.extend(find_images(DFU_DIR))
    isic = find_images(ISIC_DIR, limit=800)
    paths.extend(isic)
    random.shuffle(paths)
    paths = paths[:max_total]

    X, Y = [], []
    for i, p in enumerate(paths):
        img = load_img(p)
        if img is None: continue
        mask = make_mask(img)
        X.append(img); Y.append(mask)
        if (i+1) % 200 == 0:
            log.info(f"  Loaded {i+1}/{len(paths)}…")

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    log.info(f"  ✅ Loaded {len(X)} image-mask pairs — X:{X.shape} Y:{Y.shape}")
    return X, Y

def preload_clf(stage_limit=None, isic_limit=1200):
    """Load classification data into arrays."""
    log.info("Pre-loading classification data…")

    # Stage data
    stage_imgs, stage_lbls = [], []
    for i, d in enumerate(STAGE_DIRS):
        imgs = find_images(d, limit=stage_limit)
        log.info(f"  Stage {i+1}: {len(imgs)} images")
        for p in imgs:
            img = load_img(p)
            if img is not None:
                stage_imgs.append(img)
                stage_lbls.append(i)

    # Type data  
    dfu_imgs = find_images(DFU_DIR)
    log.info(f"  DFU: {len(dfu_imgs)}")
    dfu_X = []
    for p in dfu_imgs:
        img = load_img(p)
        if img is not None: dfu_X.append(img)

    isic_paths = find_images(ISIC_DIR, limit=isic_limit)
    log.info(f"  ISIC: {len(isic_paths)}")
    isic_X = []
    for i, p in enumerate(isic_paths):
        img = load_img(p)
        if img is not None: isic_X.append(img)
        if (i+1) % 300 == 0:
            log.info(f"    ISIC loaded {i+1}/{len(isic_paths)}…")

    return (np.array(stage_imgs, dtype=np.float32),
            np.array(stage_lbls, dtype=np.int32),
            np.array(dfu_X,      dtype=np.float32),
            np.array(isic_X,     dtype=np.float32))

# ── augmented dataset ─────────────────────────────────────────────────────────
def make_aug_dataset(X, Y=None, batch=8, shuffle=True):
    """Create tf.data pipeline with augmentation from in-memory arrays."""
    if Y is not None:
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
    else:
        ds = tf.data.Dataset.from_tensor_slices(X)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(X)), seed=SEED)

    def aug_seg(img, mask):
        # Random flip
        img  = tf.image.random_flip_left_right(img,  seed=SEED)
        mask = tf.image.random_flip_left_right(mask, seed=SEED)
        img  = tf.image.random_flip_up_down(img,  seed=SEED)
        mask = tf.image.random_flip_up_down(mask, seed=SEED)
        # Random brightness / contrast
        img  = tf.image.random_brightness(img, 0.2)
        img  = tf.image.random_contrast(img, 0.8, 1.2)
        img  = tf.clip_by_value(img, 0, 1)
        return img, mask

    def aug_clf(img, lbl):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0, 1)
        return img, lbl

    if Y is not None:
        ds = ds.map(aug_seg, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(aug_clf, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# ── losses ────────────────────────────────────────────────────────────────────
def dice_loss(yt, yp, s=1e-6):
    yt = tf.keras.backend.flatten(tf.cast(yt, tf.float32))
    yp = tf.keras.backend.flatten(yp)
    i  = tf.keras.backend.sum(yt * yp)
    return 1-(2*i+s)/(tf.keras.backend.sum(yt)+tf.keras.backend.sum(yp)+s)

def combo_loss(yt, yp):
    bce = tf.keras.losses.binary_crossentropy(tf.cast(yt, tf.float32), yp)
    return 0.4*tf.reduce_mean(bce) + 0.6*dice_loss(yt, yp)

def dice_coeff(yt, yp, s=1e-6):
    yt = tf.keras.backend.flatten(tf.cast(yt, tf.float32))
    yp = tf.keras.backend.flatten(yp)
    i  = tf.keras.backend.sum(yt * yp)
    return (2*i+s)/(tf.keras.backend.sum(yt)+tf.keras.backend.sum(yp)+s)

# ── segmentation model ────────────────────────────────────────────────────────
def build_unet():
    inp  = tf.keras.layers.Input(shape=(SZ, SZ, 3))
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inp)

    if PRETRAINED.exists():
        try:
            old = tf.keras.models.load_model(str(PRETRAINED), compile=False)
            n = 0
            for lyr in base.layers:
                try:
                    base.get_layer(lyr.name).set_weights(
                        old.get_layer(lyr.name).get_weights()); n+=1
                except: pass
            log.info(f"  Warm-started {n} encoder layers")
        except Exception as e:
            log.warning(f"  Warm-start failed: {e}")

    base.trainable = False

    L  = tf.keras.layers
    s1 = base.get_layer("conv1_relu").output
    s2 = base.get_layer("conv2_block3_out").output
    s3 = base.get_layer("conv3_block4_out").output
    s4 = base.get_layer("conv4_block6_out").output
    bn = base.get_layer("conv5_block3_out").output

    def cb(x, f):
        for _ in range(2):
            x = L.Conv2D(f, 3, padding="same")(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
        return x

    def up(x, skip, f):
        x = L.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = L.Concatenate()([x, skip])
        return cb(x, f)

    d1  = up(bn, s4, 512)
    d2  = up(d1, s3, 256)
    d3  = up(d2, s2, 128)
    d4  = up(d3, s1,  64)
    x   = L.Conv2DTranspose(32, 2, strides=2, padding="same")(d4)
    x   = cb(x, 32)
    out = L.Conv2D(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inp, out, name="unet256"), base

# ── classifier model ──────────────────────────────────────────────────────────
def build_clf(n_cls, name):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(SZ, SZ, 3), include_top=False, weights="imagenet")
    base.trainable = False
    L = tf.keras.layers
    x = L.GlobalAveragePooling2D()(base.output)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(0.4)(x)
    x = L.Dense(128, activation="relu")(x)
    out = L.Dense(n_cls, activation="softmax")(x)
    return tf.keras.Model(base.input, out, name=name)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    Path("models").mkdir(exist_ok=True)
    t0 = time.time()

    log.info("="*55)
    log.info("WoundAI v4  –  Quick Retrain (in-memory)")
    log.info("="*55)

    # ── 1. SEGMENTATION ───────────────────────────────────────────────────────
    log.info("\n[1/3] SEGMENTATION  256×256")
    X_seg, Y_seg = preload_seg(max_total=3000)

    split = int(len(X_seg) * 0.85)
    idx   = np.random.permutation(len(X_seg))
    tr_i, va_i = idx[:split], idx[split:]

    Xtr, Ytr = X_seg[tr_i], Y_seg[tr_i]
    Xva, Yva = X_seg[va_i], Y_seg[va_i]
    log.info(f"  Train: {len(Xtr)}  Val: {len(Xva)}")

    ds_tr = make_aug_dataset(Xtr, Ytr, batch=6)
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, Yva)).batch(6)

    model, base = build_unet()

    cb_seg = [
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT_SEG), monitor="val_dice_coeff",
            save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coeff", patience=6,
            mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1),
    ]

    log.info("  Phase A: frozen encoder…")
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss=combo_loss, metrics=[dice_coeff])
    model.fit(ds_tr, validation_data=ds_va,
              epochs=15, callbacks=cb_seg)

    log.info("  Phase B: unfreeze top 60 layers…")
    base.trainable = True
    for lyr in base.layers[:-60]:
        lyr.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(4e-5),
                  loss=combo_loss, metrics=[dice_coeff])
    model.fit(ds_tr, validation_data=ds_va,
              epochs=15, callbacks=cb_seg)

    log.info(f"  ✅ Segmentation saved → {OUT_SEG}")
    del X_seg, Y_seg, Xtr, Ytr, Xva, Yva  # free RAM

    # ── 2. STAGE CLASSIFIER ───────────────────────────────────────────────────
    log.info("\n[2/3] STAGE CLASSIFIER  (Stage 1–4)")
    stage_X, stage_y, dfu_X, isic_X = preload_clf()

    # One-hot encode
    oh_stage = np.eye(4, dtype=np.float32)[stage_y]

    idx   = np.random.permutation(len(stage_X))
    split = int(len(stage_X) * 0.8)
    tr_i, va_i = idx[:split], idx[split:]

    ds_tr_s = make_aug_dataset(stage_X[tr_i], oh_stage[tr_i], batch=16)
    ds_va_s = tf.data.Dataset.from_tensor_slices(
                  (stage_X[va_i], oh_stage[va_i])).batch(16)

    clf_s = build_clf(4, "stage_clf")
    clf_s.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    cb_s = [
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT_STAGE), monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            mode="max", restore_best_weights=True),
    ]

    clf_s.fit(ds_tr_s, validation_data=ds_va_s, epochs=25, callbacks=cb_s)

    # Fine-tune
    clf_s.layers[0].trainable = True
    clf_s.compile(optimizer=tf.keras.optimizers.Adam(2e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    clf_s.fit(ds_tr_s, validation_data=ds_va_s, epochs=10, callbacks=cb_s)
    log.info(f"  ✅ Stage classifier saved → {OUT_STAGE}")

    # ── 3. TYPE CLASSIFIER ────────────────────────────────────────────────────
    log.info("\n[3/3] TYPE CLASSIFIER  (DFU / Ulcer / Skin Lesion)")

    type_X = np.concatenate([
        dfu_X,
        stage_X,
        isic_X
    ], axis=0)
    type_y = np.array(
        [0]*len(dfu_X) + [1]*len(stage_X) + [2]*len(isic_X),
        dtype=np.int32)
    oh_type = np.eye(3, dtype=np.float32)[type_y]

    idx   = np.random.permutation(len(type_X))
    split = int(len(type_X) * 0.8)
    tr_i, va_i = idx[:split], idx[split:]

    ds_tr_t = make_aug_dataset(type_X[tr_i], oh_type[tr_i], batch=16)
    ds_va_t = tf.data.Dataset.from_tensor_slices(
                  (type_X[va_i], oh_type[va_i])).batch(16)

    clf_t = build_clf(3, "type_clf")
    clf_t.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    cb_t = [
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT_TYPE), monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            mode="max", restore_best_weights=True),
    ]

    clf_t.fit(ds_tr_t, validation_data=ds_va_t, epochs=25, callbacks=cb_t)

    clf_t.layers[0].trainable = True
    clf_t.compile(optimizer=tf.keras.optimizers.Adam(2e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    clf_t.fit(ds_tr_t, validation_data=ds_va_t, epochs=10, callbacks=cb_t)

    with open("models/wound_type_classes.json", "w") as f:
        json.dump(["DFU (Diabetic Foot Ulcer)",
                   "Pressure/Venous Ulcer",
                   "Skin Lesion"], f)
    log.info(f"  ✅ Type classifier saved → {OUT_TYPE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    log.info(f"\n✅ All done in {elapsed:.1f} minutes")
    log.info("\nModels saved:")
    for p in [OUT_SEG, OUT_STAGE, OUT_TYPE]:
        exists = "✅" if p.exists() else "❌"
        size   = f"{p.stat().st_size/1e6:.0f}MB" if p.exists() else "missing"
        log.info(f"  {exists} {p}  ({size})")
    log.info("\nNext: redeploy to Cloud Run")
    log.info("  gcloud run deploy wound-api --source . --region us-central1 \\")
    log.info("    --memory 4Gi --cpu 2 --timeout 600 --allow-unauthenticated")

if __name__ == "__main__":
    main()