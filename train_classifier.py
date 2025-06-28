import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import logging
from PIL import Image
import glob
import hashlib
from sklearn.metrics import confusion_matrix
import seaborn as sns

# === Mixed Precision ===
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/nadiajelani/projects/wound-segmentation/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Config ===
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced for Apple M3
EPOCHS = 30  # Increased for better convergence
DATASET_PATH = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset'
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VAL_DIR = os.path.join(DATASET_PATH, 'validation')
MODEL_SAVE_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/wound_classifier.h5'
CHECKPOINT_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/checkpoint_{epoch:02d}.h5'
CLASS_NAMES = ['non-wound', 'wound']
EXPECTED_TRAIN_COUNTS = {'non-wound': 8012, 'wound': 13218}
EXPECTED_VAL_COUNTS = {'non-wound': 2003, 'wound': 3305}

# === Verify Directories ===
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
for dir_path in [TRAIN_DIR, VAL_DIR]:
    if not os.path.exists(dir_path):
        logger.error(f"Directory missing: {dir_path}")
        raise FileNotFoundError(f"Directory {dir_path} does not exist")

# === Check Dataset Integrity ===
def check_dataset_integrity(directory, class_names=CLASS_NAMES, expected_counts=None):
    corrupted_files, duplicates, subdirs, small_files = [], [], [], []
    file_hashes, file_names = {}, set()
    class_counts = {c: 0 for c in class_names}
    sample_files = {c: [] for c in class_names}
    expected_subdirs = [os.path.join(directory, c) for c in class_names]
    
    for subdir in expected_subdirs:
        if not os.path.exists(subdir):
            logger.warning(f"⚠️ Creating missing subdirectory: {subdir}")
            os.makedirs(subdir)
        else:
            logger.info(f"Found subdirectory: {subdir}")
    
    for root, dirs, files in os.walk(directory):
        if root != directory and root not in expected_subdirs:
            subdirs.append(root)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(image_files)
        for class_name in class_names:
            if class_name in os.path.basename(root).lower():
                class_counts[class_name] += total_files
                if len(sample_files[class_name]) < 5:
                    sample_files[class_name].extend([os.path.join(root, f) for f in image_files[:5-len(sample_files[class_name])]])
        for file_name in image_files:
            file_path = os.path.join(root, file_name)
            try:
                if os.path.getsize(file_path) < 1024:
                    small_files.append(file_path)
                    continue
                if file_name in file_names:
                    duplicates.append((file_path, f"Duplicate name: {file_name}"))
                file_names.add(file_name)
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in file_hashes:
                    duplicates.append((file_path, file_hashes[file_hash]))
                else:
                    file_hashes[file_hash] = file_path
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                logger.error(f"❌ Corrupted file: {file_path} - {e}")
                corrupted_files.append(file_path)
    
    logger.info(f"Total images in {directory}: {sum(class_counts.values())}")
    logger.info(f"Class counts: {class_counts}")
    for class_name, files in sample_files.items():
        logger.info(f"Sample files for {class_name}: {files}")
    if duplicates:
        logger.warning(f"⚠️ Duplicates: {len(duplicates)} files")
        for dup_path, reason in duplicates:
            logger.warning(f"Duplicate: {dup_path} ({reason})")
    if subdirs:
        logger.warning(f"⚠️ Nested subdirs: {subdirs}")
    if small_files:
        logger.warning(f"⚠️ Small files (<1KB): {len(small_files)}")
    if corrupted_files:
        logger.warning(f"⚠️ Corrupted files: {len(corrupted_files)}")
    if expected_counts:
        expected_total = sum(expected_counts.values())
        if sum(class_counts.values()) != expected_total:
            logger.warning(f"⚠️ Count mismatch: Expected {expected_total}, found {sum(class_counts.values())}")
        for class_name, count in expected_counts.items():
            if class_counts.get(class_name, 0) != count:
                logger.warning(f"⚠️ {class_name} mismatch: Expected {count}, found {class_counts.get(class_name, 0)}")
    
    return corrupted_files, duplicates, subdirs, small_files, class_counts

logger.info("Checking dataset integrity...")
train_corrupted, train_duplicates, train_subdirs, train_small, train_class_counts = check_dataset_integrity(TRAIN_DIR, expected_counts=EXPECTED_TRAIN_COUNTS)
val_corrupted, val_duplicates, val_subdirs, val_small, val_class_counts = check_dataset_integrity(VAL_DIR, expected_counts=EXPECTED_VAL_COUNTS)

# === Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increased for diversity
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

try:
    logger.info(f"Loading training data from {TRAIN_DIR}")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=CLASS_NAMES,
        shuffle=True
    )
    logger.info(f"Loading validation data from {VAL_DIR}")
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=CLASS_NAMES,
        shuffle=False
    )
except Exception as e:
    logger.error(f"❌ Failed to load data: {e}")
    raise

logger.info(f"Found {train_generator.samples} training images, {val_generator.samples} validation images")
if train_generator.samples != sum(train_class_counts.values()) or val_generator.samples != sum(val_class_counts.values()):
    logger.warning(f"⚠️ Mismatch: flow_from_directory reports {train_generator.samples}/{val_generator.samples}, found {sum(train_class_counts.values())}/{sum(val_class_counts.values())}")
logger.info(f"Class indices: {train_generator.class_indices}")

# === Check Training Feasibility ===
if train_generator.samples == 0 or val_generator.samples == 0:
    logger.error("❌ Cannot train: No images found")
    raise ValueError("No images found")

# === Class Weights ===
class_totals = np.bincount(train_generator.classes)
total = class_totals.sum()
class_weights = {i: total / (2 * count) for i, count in enumerate(class_totals) if count > 0}
logger.info(f"Class counts: {class_totals}")
logger.info(f"Class weights: {class_weights}")

# === Visualize Sample Images ===
def visualize_sample_images(generator, directory, num_samples=3):
    try:
        x, y = next(generator)
        class_names = {v: k for k, v in generator.class_indices.items()}
        files = generator.filenames[:num_samples]
        plt.figure(figsize=(12, 4))
        for i in range(min(num_samples, len(x))):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(x[i])
            plt.title(f"Class: {class_names[int(y[i])]}\n{os.path.basename(files[i])}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), f'sample_images_{os.path.basename(directory)}.png'))
        plt.close()
        return files, [class_names[int(y[i])] for i in range(min(num_samples, len(x)))]
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        return [], []

if train_generator.samples > 0 and val_generator.samples > 0:
    logger.info("Visualize sample images? (y/n)")
    if input().lower() == 'y':
        logger.info("Visualizing training images...")
        train_files, train_labels = visualize_sample_images(train_generator, TRAIN_DIR)
        logger.info(f"Training samples: {train_files} ({train_labels})")
        logger.info("Visualizing validation images...")
        val_files, val_labels = visualize_sample_images(val_generator, VAL_DIR)
        logger.info(f"Validation samples: {val_files} ({val_labels})")

# === Model ===
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
# Unfreeze later layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# === Load Weights ===
try:
    latest_checkpoint = max(glob.glob(os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'checkpoint_*.h5')), key=os.path.getctime, default=None)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        logger.info(f"✅ Loaded checkpoint: {latest_checkpoint}")
    else:
        logger.info("No checkpoint found")
except Exception as e:
    logger.warning(f"⚠️ Failed to load checkpoint: {e}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5, clipnorm=1.0),  # Lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# === Callbacks ===
class ConfusionMatrixCallback(Callback):
    def __init__(self, val_gen):
        super().__init__()
        self.val_gen = val_gen
    def on_epoch_end(self, epoch, logs=None):
        self.val_gen.reset()
        y_true, y_pred = [], []
        try:
            steps = min(len(self.val_gen), 100)
            for _ in range(steps):
                x, y = next(self.val_gen)
                pred = self.model.predict(x, verbose=0)
                y_true.extend(y)
                y_pred.extend((pred > 0.5).astype(np.float32).flatten())
            cm = confusion_matrix(y_true, y_pred)
            logger.info(f"Epoch {epoch + 1} Confusion Matrix:\n{cm}")
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            plt.title(f'Epoch {epoch + 1} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), f'cm_epoch_{epoch+1}.png'))
            plt.close()
        except Exception as e:
            logger.error(f"❌ Confusion matrix failed: {e}")

callbacks = [
    ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_auc', mode='max', verbose=1),
    EarlyStopping(monitor='val_auc', patience=7, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_auc', patience=3, factor=0.2, min_lr=1e-6, verbose=1),
    ConfusionMatrixCallback(val_generator)
]

# === Train ===
logger.info("Starting training...")
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks
    )
except Exception as e:
    logger.error(f"❌ Training failed: {e}")
    raise

# === Save Model ===
model.save(MODEL_SAVE_PATH)
logger.info(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# === Evaluate ===
logger.info("Evaluating on validation set...")
val_loss, val_acc, val_auc, val_prec, val_rec = model.evaluate(val_generator)
logger.info(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

# === Visualize Predictions ===
def visualize_predictions(model, generator, directory, num_samples=3):
    try:
        x, y = next(generator)
        predictions = model.predict(x[:num_samples])
        predictions = (predictions > 0.5).astype(np.float32)
        class_names = {v: k for k, v in generator.class_indices.items()}
        files = generator.filenames[:num_samples]
        plt.figure(figsize=(10, 5))
        for i in range(num_samples):
            plt.subplot(num_samples, 2, 2*i+1)
            plt.imshow(x[i])
            plt.title(f"True: {class_names[int(y[i])]}\n{os.path.basename(files[i])}")
            plt.axis('off')
            plt.subplot(num_samples, 2, 2*i+2)
            plt.imshow(x[i])
            plt.title(f"Predicted: {class_names[int(predictions[i][0])]}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'predictions.png'))
        plt.close()
    except Exception as e:
        logger.error(f"❌ Prediction visualization failed: {e}")

logger.info("Visualizing predictions...")
visualize_predictions(model, val_generator, VAL_DIR)

# === Plot Training History ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'training_history.png'))
plt.close()
logger.info("Training history plot saved to training_history.png")