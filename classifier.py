import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import logging

# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIG ===
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced to prevent memory issues
EPOCHS = 50  # Increased for better convergence
DATASET_PATH = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset"
MODEL_SAVE_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/wound_classifier_optimized.keras"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "validation")

# === Verify Directories ===
for dir_path in [TRAIN_DIR, VAL_DIR]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

# === DATA GENERATORS ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Enhanced augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

logger.info(f"Found {train_gen.samples} training images and {val_gen.samples} validation images.")
logger.info(f"Class indices: {train_gen.class_indices}")

# === Compute Class Weights ===
class_counts = np.bincount(train_gen.classes)
total_samples = sum(class_counts)
class_weights = {
    0: total_samples / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
    1: total_samples / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
}
logger.info(f"Class weights: {class_weights}")

# === BUILD MODEL ===
logger.info("Building model...")
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Freeze initially

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Added layer
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# === LOAD WEIGHTS ===
try:
    model.load_weights(MODEL_SAVE_PATH.replace('.keras', '.h5'))  # Support legacy .h5
    logger.info(f"✅ Loaded weights from {MODEL_SAVE_PATH.replace('.keras', '.h5')}")
except Exception as e:
    logger.warning(f"⚠️ No weights found, starting from scratch: {e}")

# === COMPILE MODEL ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# === CALLBACKS ===
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False  # Save full model
)

# === TRAIN ===
logger.info("Starting training...")
try:
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
except Exception as e:
    logger.error(f"❌ Training failed: {e}")
    raise

# === SAVE MODEL ===
model.save(MODEL_SAVE_PATH)
logger.info(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# === EVALUATE ===
logger.info("Evaluating on validation set...")
val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(val_gen)
logger.info(f"Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

# === VISUALIZE PREDICTIONS ===
def visualize_predictions(model, generator, num_samples=5):
    x, y = next(generator)
    predictions = model.predict(x[:num_samples])
    predictions = (predictions > 0.5).astype(np.float32)
    class_names = {v: k for k, v in generator.class_indices.items()}
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(x[i])
        plt.title(f"True: {class_names[int(y[i])]}")
        plt.subplot(1, 2, 2)
        plt.imshow(x[i])
        plt.title(f"Predicted: {class_names[int(predictions[i][0])]}")
        plt.show()

logger.info("Visualizing predictions...")
visualize_predictions(model, val_gen)

# === FINE-TUNE (Optional) ===
logger.info("Fine-tuning top ResNet50 layers...")
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Fine-tune last 10 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

logger.info("Starting fine-tuning...")
try:
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        initial_epoch=history.epoch[-1] + 1 if history.epoch else 0,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
except Exception as e:
    logger.error(f"❌ Fine-tuning failed: {e}")
    raise

model.save(MODEL_SAVE_PATH)
logger.info(f"✅ Fine-tuned model saved to: {MODEL_SAVE_PATH}")

# === FINAL EVALUATION ===
val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(val_gen)
logger.info(f"Final Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
visualize_predictions(model, val_gen)