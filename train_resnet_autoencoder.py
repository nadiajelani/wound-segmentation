
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
import logging
import os
import matplotlib.pyplot as plt
from collections import Counter
import pickle

print("Imports completed")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("Logging set up")

# Enable mixed precision
set_global_policy('mixed_float16')

print("Mixed precision enabled")

# Verify GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"GPU detected: {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    logger.info("No GPU detected, training will be slower on CPU.")

print("GPU check completed")

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16  # Reduced to lower memory usage
PRETRAIN_EPOCHS = 5
INITIAL_EPOCHS = 10
TOTAL_EPOCHS = 20
TRAIN_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset/train'
VALID_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset/validation'
CHECKPOINT_DIR = '/Users/nadiajelani/projects/wound-segmentation/models/checkpoints'
MODEL_SAVE_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/wound_classifier_simclr.keras'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'wound_classifier_simclr_{epoch:02d}_{val_accuracy:.4f}.keras')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Configuration set")

# Function to load and preprocess images
def load_and_preprocess_image(file_path, label=None):
    try:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = img / 255.0
        img = tf.ensure_shape(img, [IMG_HEIGHT, IMG_WIDTH, 3])
        if label is None:
            return img
        return img, label
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {str(e)}")
        invalid_img = tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
        if label is None:
            return invalid_img
        return invalid_img, label

# Function to check if an image is valid (non-zero)
def is_valid_image(img):
    return tf.reduce_sum(img) > 0.0

# Function to check if an image-label pair is valid
def is_valid_image_with_label(img, label):
    return tf.reduce_sum(img) > 0.0

# SimCLR Augmentation
def simclr_augmentation(img):
    # Random crop and resize
    img = tf.image.random_crop(img, size=[IMG_HEIGHT - 20, IMG_WIDTH - 20, 3])
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # Random flip
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # Random brightness and contrast
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    # Ensure pixel values are in [0, 1]
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

# Create a pair of augmented views for SimCLR
def create_simclr_pair(img):
    img1 = simclr_augmentation(img)
    img2 = simclr_augmentation(img)
    return img1, img2

# Create a tf.data.Dataset for SimCLR pretraining
def create_pretrain_dataset(directory):
    print("Starting create_pretrain_dataset")
    file_paths = []
    for root, _, files in os.walk(directory):
        print(f"Scanning directory: {root}")
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_paths.append(os.path.join(root, file))
    
    print(f"Found {len(file_paths)} files before filtering")
    logger.info(f"Found {len(file_paths)} images for pretraining in {directory}")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.map(
        lambda path: load_and_preprocess_image(path),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.filter(lambda img: tf.reduce_sum(img) > 0.0)
    dataset = dataset.map(
        lambda img: create_simclr_pair(img),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    for batch in dataset.take(1):
        img1, img2 = batch
        tf.debugging.assert_all_finite(img1, 'NaNs or Infs in pretraining batch (img1).')
        tf.debugging.assert_all_finite(img2, 'NaNs or Infs in pretraining batch (img2).')

    logger.info(f"Pretraining dataset ready: {len(file_paths)} files (filtered invalids)")
    return dataset, len(file_paths)

# Create a tf.data.Dataset for supervised training
def create_supervised_dataset(directory, shuffle=True):
    print("Starting create_supervised_dataset")
    file_paths = []
    labels = []

    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_paths.append(os.path.join(class_dir, file))
                labels.append(class_to_idx[class_name])

    logger.info(f"Found {len(file_paths)} images in {directory} with classes: {class_names}")

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)

    if shuffle:
        dataset = dataset.repeat()

    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image(path, tf.cast(label, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.filter(lambda img, label: tf.reduce_sum(img) > 0.0)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    for images, labels in dataset.take(1):
        tf.debugging.assert_all_finite(images, 'NaNs or Infs found in image batch.')
        tf.debugging.assert_all_finite(labels, 'NaNs or Infs found in label batch.')

    logger.info(f"Supervised dataset ({directory}) has valid batches after filtering")
    return dataset, len(file_paths), class_to_idx

# SimCLR Contrastive Loss
def contrastive_loss(projections_1, projections_2, temperature=0.1):
    batch_size = tf.shape(projections_1)[0]
    projections_1 = tf.math.l2_normalize(projections_1, axis=1)
    projections_2 = tf.math.l2_normalize(projections_2, axis=1)
    
    sim_matrix = tf.matmul(projections_1, projections_2, transpose_b=True) / temperature
    
    labels = tf.range(batch_size)
    labels = tf.cast(labels, tf.int32)
    
    loss_1 = tf.keras.losses.sparse_categorical_crossentropy(labels, sim_matrix, from_logits=True)
    loss_2 = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(sim_matrix), from_logits=True)
    
    return 0.5 * (loss_1 + loss_2)

# Build SimCLR Model
def build_simclr_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    projections = Dense(128, activation=None)(x)
    model = Model(inputs=base_model.input, outputs=projections)
    return model

# Pretraining with SimCLR
print("Creating pretraining dataset")
pretrain_dataset, num_images = create_pretrain_dataset(TRAIN_DIR)

print("Building SimCLR model")
simclr_model = build_simclr_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step_simclr(img1, img2):
    with tf.GradientTape() as tape:
        proj1 = simclr_model(img1, training=True)
        proj2 = simclr_model(img2, training=True)
        loss = tf.reduce_mean(contrastive_loss(proj1, proj2))
    gradients = tape.gradient(loss, simclr_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, simclr_model.trainable_variables))
    return loss

logger.info("Starting SimCLR pretraining...")
steps_per_epoch = max(1, num_images // BATCH_SIZE)
for epoch in range(PRETRAIN_EPOCHS):
    logger.info(f"Epoch {epoch + 1}/{PRETRAIN_EPOCHS}")
    step = 0
    for img1, img2 in pretrain_dataset.take(steps_per_epoch):
        loss = train_step_simclr(img1, img2)
        step += 1
        if step % 100 == 0:
            logger.info(f"Step {step}/{steps_per_epoch}, Loss: {loss:.4f}")
    logger.info(f"Epoch {epoch + 1} completed, Loss: {loss:.4f}")

# Save the pretrained base model
base_model = simclr_model.get_layer(index=0)
base_model.save('/Users/nadiajelani/projects/wound-segmentation/models/pretrained_simclr_base.keras')
logger.info("SimCLR pretraining completed, base model saved.")

# Supervised training
print("Creating supervised training dataset")
train_dataset, num_train_images, class_to_idx = create_supervised_dataset(TRAIN_DIR, shuffle=True)
val_dataset, num_val_images, _ = create_supervised_dataset(VALID_DIR, shuffle=False)

for images, labels in train_dataset.take(1):
    logger.info(f"Training batch shape: {images.shape}, Label shape: {labels.shape}")
    tf.debugging.assert_all_finite(images, 'Found NaNs or Infs in training batch.')
    tf.debugging.assert_all_finite(labels, 'Found NaNs or Infs in training labels.')
    logger.info("Training batch passed finite check.")

for images, labels in val_dataset.take(1):
    logger.info(f"Validation batch shape: {images.shape}, Label shape: {labels.shape}")
    tf.debugging.assert_all_finite(images, 'Found NaNs or Infs in validation batch.')
    tf.debugging.assert_all_finite(labels, 'Found NaNs or Infs in validation labels.')
    logger.info("Validation batch passed finite check.")

logger.info(f"Classes found: {class_to_idx}")
if len(class_to_idx) != 2:
    raise ValueError(f"Expected 2 classes (wound, non_wound), but found {len(class_to_idx)} classes: {class_to_idx}")

train_labels = []
for _, label in train_dataset.unbatch().take(num_train_images):
    train_labels.append(int(label.numpy()))
class_counts = Counter(train_labels)
logger.info(f"Class distribution in training set: {class_counts}")
class_weights = {
    0: len(train_labels) / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
    1: len(train_labels) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
}
logger.info(f"Class weights: {class_weights}")

train_steps_per_epoch = max(1, num_train_images // BATCH_SIZE)
val_steps_per_epoch = max(1, num_val_images // BATCH_SIZE)
logger.info(f"Train steps per epoch: {train_steps_per_epoch}")
logger.info(f"Validation steps per epoch: {val_steps_per_epoch}")

# Load pretrained base model and build classifier
base_model = tf.keras.models.load_model('/Users/nadiajelani/projects/wound-segmentation/models/pretrained_simclr_base.keras')
for layer in base_model.layers[:150]:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
predictions = Activation('sigmoid', dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=predictions)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()],
              jit_compile=True)

checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

class LossMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') > 10.0 or np.isnan(logs.get('loss')):
            logger.warning(f"High loss detected ({logs.get('loss'):.4f}) at epoch {epoch}. Stopping training.")
            self.model.stop_training = True

logger.info("Starting supervised training...")
try:
    history_initial = model.fit(
        train_dataset,
        steps_per_epoch=train_steps_per_epoch,
        epochs=INITIAL_EPOCHS,
        validation_data=val_dataset,
        validation_steps=val_steps_per_epoch,
        callbacks=[checkpoint, early_stopping, reduce_lr, LossMonitor()],
        class_weight=class_weights,
        initial_epoch=0
    )
except KeyboardInterrupt:
    logger.info("Training interrupted. Saving model...")
    model.save(MODEL_SAVE_PATH)
    raise

for layer in base_model.layers[:150]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()],
              jit_compile=True)

try:
    history_finetune = model.fit(
        train_dataset,
        steps_per_epoch=train_steps_per_epoch,
        epochs=TOTAL_EPOCHS,
        validation_data=val_dataset,
        validation_steps=val_steps_per_epoch,
        callbacks=[checkpoint, early_stopping, reduce_lr, LossMonitor()],
        class_weight=class_weights,
        initial_epoch=INITIAL_EPOCHS
    )
except KeyboardInterrupt:
    logger.info("Fine-tuning interrupted. Saving model...")
    model.save(MODEL_SAVE_PATH)
    raise

full_history = {k: history_initial.history[k] + (history_finetune.history[k] if 'history_finetune' in locals() and k in history_finetue.history else []) for k in history_initial.history.keys()}

val_dataset = val_dataset.unbatch().batch(BATCH_SIZE)
val_predictions = model.predict(val_dataset, steps=val_steps_per_epoch, verbose=0)
val_labels = np.concatenate([y for _, y in val_dataset.unbatch().take(num_val_images)], axis=0)

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

model.save(MODEL_SAVE_PATH)
logger.info(f"Final model saved to {MODEL_SAVE_PATH}")

with open('training_history.pkl', 'wb') as f:
    pickle.dump(full_history, f)
logger.info("Training history saved to training_history.pkl")

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
