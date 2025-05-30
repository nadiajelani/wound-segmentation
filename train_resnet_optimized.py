
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset/train'
VALID_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset/validation'
MODEL_SAVE_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/wound_classifier_optimized.h5'

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)

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

# Build model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
for layer in base_model.layers[:-10]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile with binary_crossentropy to isolate loss issue
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Initial training
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

# Fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate and optimize threshold
val_images, val_labels = next(validation_generator)
val_predictions = model.predict(val_images)
val_predictions_binary = (val_predictions > 0.7).astype(int)

auc = roc_auc_score(val_labels, val_predictions)
conf_matrix = confusion_matrix(val_labels, val_predictions_binary)
precision, recall, thresholds = precision_recall_curve(val_labels, val_predictions)
optimal_threshold = thresholds[np.argmax(2 * (precision * recall) / (precision + recall))]
logger.info(f"Validation AUC: {auc:.4f}")
logger.info(f"Confusion Matrix:\n{conf_matrix}")
logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")

# Save model
model.save(MODEL_SAVE_PATH)
logger.info(f"Model saved to {MODEL_SAVE_PATH}")
