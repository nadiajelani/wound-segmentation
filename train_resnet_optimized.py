
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
import logging
import os
from collections import Counter
import pickle
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/Users/nadiajelani/projects/wound-segmentation/wound_progress.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set dtype policy to mixed precision for memory efficiency
tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Adjusted for GPU memory

# Verify GPU device
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logger.info(f"GPU detected: {physical_devices}")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    logger.warning("No GPU detected, falling back to CPU. Performance may be reduced.")

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32  # Increased to 32 for SimCLR, adjust if memory issues arise
PRETRAIN_EPOCHS = 50  # Increased for better pretraining
INITIAL_EPOCHS = 10
TOTAL_EPOCHS = 30  # Extended for fine-tuning
TRAIN_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset/train'
VALID_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset/validation'
CHECKPOINT_DIR = '/Users/nadiajelani/projects/wound-segmentation/models/checkpoints'
MODEL_SAVE_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/wound_classifier_optimized.keras'
PRETRAIN_SAVE_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/pretrained_base_model.keras'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'wound_classifier_{epoch:02d}_{val_accuracy:.4f}.keras')
PRETRAIN_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'pretrain_wound_classifier_{epoch:02d}.keras')
HISTORY_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/training_history.pkl'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Data augmentation for SimCLR
def simclr_augmentation():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased for stronger augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Wider range
        fill_mode='nearest',
        preprocessing_function=lambda x: tf.image.random_contrast(x, 0.8, 1.2)  # Added contrast
    )

# Data generators
simclr_datagen = simclr_augmentation()
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Combine train for SimCLR pretraining
all_generator = simclr_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=True
)
all_dataset = tf.data.Dataset.from_generator(
    lambda: all_generator,
    output_signature=tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
).map(
    lambda x: tf.clip_by_value(x, 0.0, 1.0),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Visual check of augmented batches
logger.info("Performing visual check of augmented batches...")
for images in all_dataset.take(1):
    logger.info(f"Batch shape: {images.shape}")
    logger.info(f"Batch min/max: {np.min(images):.4f}/{np.max(images):.4f}")
    for i in range(min(4, images.shape[0])):
        plt.figure(figsize=(5, 5))
        plt.imshow(images[i])
        plt.title(f"Image {i}")
        plt.axis('off')
        plt.show()
    if np.all(images == images[0]):  # Check for identical images
        logger.error("All images in batch are identical! Check data or augmentations.")
    else:
        logger.info("Batch contains diverse images.")

# --- SimCLR Pretraining ---
@tf.function
def simclr_train_step(images, model, optimizer, temperature=0.1):
    with tf.GradientTape() as tape:
        # Generate two augmented views
        view1 = tf.image.random_flip_left_right(images)
        view1 = tf.image.random_brightness(view1, max_delta=0.1)  # Increased delta
        view1 = tf.image.random_contrast(view1, lower=0.8, upper=1.2)  # Wider range
        view1 = tf.image.random_crop(view1, size=[tf.shape(images)[0], IMG_HEIGHT, IMG_WIDTH, 3])
        view1 = tf.clip_by_value(view1, 0.0, 1.0)
        
        view2 = tf.image.random_flip_left_right(images)
        view2 = tf.image.random_brightness(view2, max_delta=0.1)
        view2 = tf.image.random_contrast(view2, lower=0.8, upper=1.2)
        view2 = tf.image.random_crop(view2, size=[tf.shape(images)[0], IMG_HEIGHT, IMG_WIDTH, 3])
        view2 = tf.clip_by_value(view2, 0.0, 1.0)
        
        h1 = model(view1, training=True)
        h2 = model(view2, training=True)
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)
        batch_size = tf.shape(h1)[0]
        labels = tf.range(batch_size)
        logits_ab = tf.matmul(h1, h2, transpose_b=True) / temperature
        logits_ba = tf.matmul(h2, h1, transpose_b=True) / temperature
        masks = tf.one_hot(tf.range(batch_size), batch_size, dtype=logits_ab.dtype)
        logits_aa = tf.matmul(h1, h1, transpose_b=True) / temperature - masks * 1e9
        logits_bb = tf.matmul(h2, h2, transpose_b=True) / temperature - masks * 1e9
        loss_a = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits_ab, from_logits=True))
        loss_b = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits_ba, from_logits=True))
        loss = (loss_a + loss_b) / 2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Pretrain only if not already done
if not os.path.exists(PRETRAIN_SAVE_PATH) and not os.path.exists(PRETRAIN_SAVE_PATH.replace('.keras', '.h5')):
    logger.info("Starting SimCLR pretraining...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    pretrain_model = Model(inputs=base_model.input, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for epoch in range(PRETRAIN_EPOCHS):
        logger.info(f"Pretraining Epoch {epoch + 1}/{PRETRAIN_EPOCHS}")
        epoch_loss = 0.0
        steps = 0
        for images in all_dataset:
            loss = simclr_train_step(images, pretrain_model, optimizer)
            epoch_loss += loss.numpy()  # Convert to numpy for logging
            steps += 1
            if steps % 10 == 0:  # Log more frequently
                logger.info(f"Step {steps}, Loss: {loss.numpy():.4f}")
        epoch_loss /= steps
        logger.info(f"Pretraining Epoch {epoch + 1} Avg Loss: {epoch_loss:.4f}")
        pretrain_model.save(PRETRAIN_CHECKPOINT_PATH.format(epoch=epoch + 1))
    pretrain_model.save(PRETRAIN_SAVE_PATH)
    logger.info(f"Pretraining completed, model saved to {PRETRAIN_SAVE_PATH}")
else:
    logger.info(f"Pretrained model found at {PRETRAIN_SAVE_PATH or PRETRAIN_SAVE_PATH.replace('.keras', '.h5')}, skipping pretraining.")
    if os.path.exists(PRETRAIN_SAVE_PATH.replace('.keras', '.h5')):
        pretrain_model = tf.keras.models.load_model(PRETRAIN_SAVE_PATH.replace('.keras', '.h5'))
        pretrain_model.save(PRETRAIN_SAVE_PATH)  # Convert to .keras
    else:
        pretrain_model = tf.keras.models.load_model(PRETRAIN_SAVE_PATH)

# --- Supervised Training (Classifier Head) ---
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)
validation_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Log and verify classes
logger.info(f"Classes found: {train_generator.class_indices}")
if len(train_generator.class_indices) != 2:
    raise ValueError(f"Expected 2 classes (wound, non_wound), but found {len(train_generator.class_indices)} classes: {train_generator.class_indices}")

# Check class balance
train_labels = [train_generator.classes[i] for i in range(len(train_generator.classes))]
class_counts = Counter(train_labels)
logger.info(f"Class distribution in training set: {class_counts}")
class_weights = {
    0: len(train_labels) / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
    1: len(train_labels) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
}
logger.info(f"Class weights: {class_weights}")

# Confirm steps per epoch
logger.info(f"Train steps per epoch: {len(train_generator)}")
logger.info(f"Validation steps per epoch: {len(validation_generator)}")

# Sanity check
for images, labels in train_generator:
    logger.info(f"Sample batch - Image shape: {images.shape}, Min/Max: {np.min(images):.4f}/{np.max(images):.4f}")
    logger.info(f"Label shape: {labels.shape}, Sample labels: {labels[:10]}")
    break

# --- Supervised Model: Attach Classifier Head ---
for layer in pretrain_model.layers:
    layer.trainable = False  # Freeze for initial supervised training

x = pretrain_model.output
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_classifier')(x)
x = Dropout(0.5, name='dropout_classifier')(x)
x = Dense(1, name='dense_output')(x)
predictions = Activation('sigmoid', dtype='float32', name='sigmoid_output')(x)
model = Model(inputs=pretrain_model.input, outputs=predictions)

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
logger.info("Compiling model...")
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    jit_compile=False
)

# Custom callback for metric logging
class CustomMetricsCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_metrics()
        logger.info(f"Starting epoch {epoch + 1}, metrics reset.")

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % 100 == 0:
            logger.info(f"Batch {batch + 1}: loss={logs.get('loss', 0.0):.4f}, accuracy={logs.get('accuracy', 0.0):.4f}, auc={logs.get('auc', 0.0):.4f}")

checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

class LossMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') > 10.0 or np.isnan(logs.get('loss')):
            logger.warning(f"High loss detected ({logs.get('loss'):.4f}) at epoch {epoch}. Stopping training.")
            self.model.stop_training = True
        model.save(CHECKPOINT_PATH.format(epoch=epoch + 1, val_accuracy=logs.get('val_accuracy', 0.0)))
        logger.info(f"Saved checkpoint to {CHECKPOINT_PATH.format(epoch=epoch + 1, val_accuracy=logs.get('val_accuracy', 0.0))}")
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(self.model.history.history, f)
        logger.info(f"Saved training history to {HISTORY_PATH}")

# Load latest checkpoint if available
latest_checkpoint = None
checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('wound_classifier_') and f.endswith('.keras') and os.path.isfile(os.path.join(CHECKPOINT_DIR, f))]
if checkpoint_files:
    try:
        latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f)))
        model = tf.keras.models.load_model(os.path.join(CHECKPOINT_DIR, latest_checkpoint))
        logger.info(f"Loaded checkpoint: {latest_checkpoint}")
        initial_epoch = int(latest_checkpoint.split('_')[2][:2])
        logger.info("Recompiling model after loading checkpoint...")
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            jit_compile=False
        )
        logger.info("Starting/resuming training from checkpoint...")
        history_initial = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=INITIAL_EPOCHS,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[checkpoint, early_stopping, reduce_lr, LossMonitor(), CustomMetricsCallback()],
            class_weight=class_weights,
            initial_epoch=initial_epoch
        )
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load checkpoint {latest_checkpoint}: {e}. Starting from initial epoch.")
        initial_epoch = 0
else:
    logger.info("No valid checkpoints found. Starting from initial epoch.")
    initial_epoch = 0
    logger.info("Starting initial training...")
    history_initial = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=INITIAL_EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr, LossMonitor(), CustomMetricsCallback()],
        class_weight=class_weights,
        initial_epoch=initial_epoch
    )

# --- Fine-tuning ---
for layer in pretrain_model.layers[:100]:
    layer.trainable = False
for layer in pretrain_model.layers[100:]:
    layer.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    jit_compile=False
)
try:
    logger.info("Starting fine-tuning...")
    history_finetune = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=TOTAL_EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr, LossMonitor(), CustomMetricsCallback()],
        class_weight=class_weights,
        initial_epoch=max(INITIAL_EPOCHS, initial_epoch)
    )
except KeyboardInterrupt:
    logger.info("Fine-tuning interrupted. Saving model...")
    model.save(MODEL_SAVE_PATH)
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump({k: history_initial.history[k] + (history_finetune.history[k] if 'history_finetune' in locals() else []) for k in history_initial.history.keys()}, f)
    logger.info(f"Training history saved to {HISTORY_PATH}")
    raise

# Combine histories
full_history = {k: history_initial.history[k] + (history_finetune.history[k] if 'history_finetune' in locals() else []) for k in history_initial.history.keys()}

# --- Evaluation ---
validation_generator.reset()
val_predictions = model.predict(validation_generator, steps=len(validation_generator), verbose=1)
val_labels = validation_generator.classes

# Metrics
auc = roc_auc_score(val_labels, val_predictions)
precision, recall, thresholds = precision_recall_curve(val_labels, val_predictions)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
val_predictions_binary = (val_predictions > optimal_threshold).astype(int)
conf_matrix = confusion_matrix(val_labels, val_predictions_binary)

logger.info(f"Validation AUC: {auc:.4f}")
logger.info(f"Confusion Matrix:\n{conf_matrix}")
logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")

# Save final model and history
model.save(MODEL_SAVE_PATH)
logger.info(f"Final model saved to {MODEL_SAVE_PATH}")
with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(full_history, f)
logger.info(f"Training history saved to {HISTORY_PATH}")

# Plot full training history
logger.info(f"Available history metrics: {full_history.keys()}")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
if 'accuracy' in full_history or 'binary_accuracy' in full_history:
    plt.plot(full_history.get('accuracy', full_history.get('binary_accuracy', [])), label='Train Acc')
if 'val_accuracy' in full_history or 'val_binary_accuracy' in full_history:
    plt.plot(full_history.get('val_accuracy', full_history.get('val_binary_accuracy', [])), label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.subplot(1, 2, 2)
if 'loss' in full_history:
    plt.plot(full_history['loss'], label='Train Loss')
if 'val_loss' in full_history:
    plt.plot(full_history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "training_plot.png"))
plt.close()
logger.info(f"Training plot saved to {os.path.join(CHECKPOINT_DIR, 'training_plot.png')}")
