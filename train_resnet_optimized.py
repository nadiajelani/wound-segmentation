
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Activation, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow_addons as tfa
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
import logging
import os
import matplotlib.pyplot as plt
from collections import Counter
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Enable mixed precision
set_global_policy('float32')

# Verify GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"GPU detected: {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    logger.warning("No GPU detected, training will be slower on CPU.")

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 10
INITIAL_EPOCHS = 10
TOTAL_EPOCHS = 20
TRAIN_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset/train'
VALID_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset/validation'
CHECKPOINT_DIR = '/Users/nadiajelani/projects/wound-segmentation/models/checkpoints'
MODEL_SAVE_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/wound_classifier_optimized.keras'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'wound_classifier_{epoch:02d}_{val_accuracy:.4f}.keras')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Data augmentation for SimCLR
def simclr_augmentation():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

# Data generators
simclr_datagen = simclr_augmentation()
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

def create_dataset_from_generator(generator):
    output_signature = (
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=output_signature
    )
    dataset = dataset.map(
        lambda x, y: (x, tf.cast(tf.expand_dims(y, axis=-1), tf.float16)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Combine train and validation for pretraining
all_generator = simclr_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,  # No labels for pretraining
    shuffle=True
)
all_dataset = tf.data.Dataset.from_generator(
    lambda: all_generator,
    output_signature=tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
).prefetch(tf.data.AUTOTUNE)

# SimCLR pretraining
def add_simclr_augmentation(images):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, max_delta=0.2)
    images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
    images = tf.image.random_crop(images, size=[tf.shape(images)[0], IMG_HEIGHT, IMG_WIDTH, 3])
    tf.debugging.check_numerics(images, "Augmented images contain NaN or Inf")
    return images

@tf.function
def simclr_train_step(images, model, optimizer, temperature=0.1):
    with tf.GradientTape() as tape:
        # Create two augmented views
        view1 = add_simclr_augmentation(images)
        view2 = add_simclr_augmentation(images)

        h1 = model(view1, training=True)
        h2 = model(view2, training=True)

        # Normalize
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)

        # Contrastive loss
        batch_size = tf.shape(h1)[0]
        labels = tf.range(batch_size)

        # Compute logits
        logits_aa = tf.matmul(h1, h1, transpose_b=True) / temperature
        logits_bb = tf.matmul(h2, h2, transpose_b=True) / temperature
        logits_ab = tf.matmul(h1, h2, transpose_b=True) / temperature
        logits_ba = tf.matmul(h2, h1, transpose_b=True) / temperature

        # Create masks with correct dtype
        masks = tf.one_hot(tf.range(batch_size), batch_size)
        masks = tf.cast(masks, dtype=logits_aa.dtype)

        # Subtract masks to zero-out self-similarity
        logits_aa -= masks * 1e9
        logits_bb -= masks * 1e9

        loss_a = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits_ab, from_logits=True))
        loss_b = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits_ba, from_logits=True))
        loss = (loss_a + loss_b) / 2
        tf.debugging.check_numerics(logits_ab, "logits_ab contains NaN or Inf")

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Base model for pretraining
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # Projection head
pretrain_model = Model(inputs=base_model.input, outputs=x)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
@tf.function
def train_step_simclr(images):
    return simclr_train_step(images, pretrain_model, optimizer)

# Pretraining loop
logger.info("Starting SimCLR pretraining...")
for epoch in range(PRETRAIN_EPOCHS):
    print(f"Pretraining Epoch {epoch + 1}/{PRETRAIN_EPOCHS}")
for images in all_dataset.take(len(all_generator)):
    loss = train_step_simclr(images)
    logger.info(f"Pretraining Epoch {epoch + 1} loss: {loss:.4f}")
pretrain_model.save('/Users/nadiajelani/projects/wound-segmentation/models/pretrained_base_model.keras')
logger.info("Pretraining completed, model saved.")

# Supervised training (reuse base model)
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

train_dataset = create_dataset_from_generator(train_generator)
val_dataset = create_dataset_from_generator(validation_generator)

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
for images, labels in train_dataset.take(1):
    logger.info(f"Sample batch - Image shape: {images.shape}, Min/Max: {tf.reduce_min(images):.4f}/{tf.reduce_max(images):.4f}")
    logger.info(f"Label shape: {labels.shape}, Sample labels: {labels.numpy().flatten()[:10]}")
    if tf.reduce_any(tf.math.is_nan(images)) or tf.reduce_any(tf.math.is_inf(images)):
        raise ValueError("NaN or Inf detected in image batch!")

# Supervised model
base_model = tf.keras.models.load_model('/Users/nadiajelani/projects/wound-segmentation/models/pretrained_base_model.keras')
for layer in base_model.layers[:-1]:  # Freeze all but the last layer
    layer.trainable = False
x = base_model.output
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
predictions = Activation('sigmoid', dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()],
              jit_compile=True)

# Callbacks
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Custom callback to monitor loss
class LossMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') > 10.0 or np.isnan(logs.get('loss')):
            logger.warning(f"High loss detected ({logs.get('loss'):.4f}) at epoch {epoch}. Stopping training.")
            self.model.stop_training = True

# Initial training
try:
    steps_per_epoch = len(train_generator)  # 38,603 / 32 ≈ 1207
    history_initial = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=INITIAL_EPOCHS,
        validation_data=val_dataset,
        validation_steps=len(validation_generator),  # 31,347 / 32 ≈ 980
        callbacks=[checkpoint, early_stopping, reduce_lr, LossMonitor()],
        class_weight=class_weights,
        initial_epoch=5
    )
except KeyboardInterrupt:
    logger.info("Training interrupted. Saving model...")
    model.save(MODEL_SAVE_PATH)
    raise

# Fine-tuning
for layer in base_model.layers[:-1]:  # Unfreeze all layers
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()],
              jit_compile=True)

try:
    history_finetune = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=TOTAL_EPOCHS,
        validation_data=val_dataset,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr, LossMonitor()],
        class_weight=class_weights,
        initial_epoch=INITIAL_EPOCHS
    )
except KeyboardInterrupt:
    logger.info("Fine-tuning interrupted. Saving model...")
    model.save(MODEL_SAVE_PATH)
    raise

# Combine histories
full_history = {k: history_initial.history[k] + (history_finetune.history[k] if 'history_finetune' in locals() and k in history_finetune.history else []) for k in history_initial.history.keys()}

# Evaluation
val_dataset = val_dataset.unbatch().batch(BATCH_SIZE)
val_predictions = model.predict(val_dataset, verbose=0)
val_labels = np.concatenate([y for _, y in val_dataset.unbatch()], axis=0)

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

# Save model
model.save(MODEL_SAVE_PATH)
logger.info(f"Final model saved to {MODEL_SAVE_PATH}")

# Save full training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(full_history, f)
logger.info("Training history saved to training_history.pkl")

# Plot full training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(full_history['accuracy'], label='Train Acc')
plt.plot(full_history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(full_history['loss'], label='Train Loss')
plt.plot(full_history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.tight_layout()
plt.savefig("training_plot.png")
plt.close()
logger.info("Training plot saved to training_plot.png")
