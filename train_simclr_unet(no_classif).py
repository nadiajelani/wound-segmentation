
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import logging
import os
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.FileHandler('/Users/nadiajelani/projects/wound-segmentation/wound_progress.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Verify GPU device
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logger.info(f"GPU detected: {physical_devices}")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    logger.warning("No GPU detected, falling back to CPU.")

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 50
TRAIN_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/train_images'
TRAIN_MASK_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/train_masks'
VALID_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_images'
VALID_MASK_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_masks'
MODEL_SAVE_PATH = '/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_wound_segmentation.keras'
CHECKPOINT_DIR = '/Users/nadiajelani/projects/wound-segmentation/models/checkpoints'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'unet_{epoch:02d}_{val_loss:.4f}.keras')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Custom U-Net decoder with SimCLR-pretrained encoder
def build_simclr_unet(input_shape=(224, 224, 3), num_classes=1, encoder_weights_path=None):
    # Load pretrained ResNet50 as encoder
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    if encoder_weights_path and os.path.exists(encoder_weights_path):
        base_model.load_weights(encoder_weights_path)
        logger.info(f"Loaded SimCLR pretrained weights from {encoder_weights_path}")
    else:
        logger.warning("No pretrained weights found, using random initialization.")

    # Encoder
    inputs = Input(shape=input_shape)
    enc1 = base_model.get_layer('conv1')(inputs)
    enc1 = base_model.get_layer('bn_conv1')(enc1)
    enc1 = base_model.get_layer('activation')(enc1)
    enc1 = base_model.get_layer('max_pooling2d_1')(enc1)
    enc2 = base_model.get_layer('conv2_block3_out')(enc1)
    enc3 = base_model.get_layer('conv3_block4_out')(enc2)
    enc4 = base_model.get_layer('conv4_block6_out')(enc3)
    enc5 = base_model.get_layer('conv5_block3_out')(enc4)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(enc5)
    up1 = Concatenate()([up1, enc4])
    up1 = Conv2D(512, 3, activation='relu', padding='same')(up1)
    up1 = Conv2D(512, 3, activation='relu', padding='same')(up1)

    up2 = UpSampling2D(size=(2, 2))(up1)
    up2 = Concatenate()([up2, enc3])
    up2 = Conv2D(256, 3, activation='relu', padding='same')(up2)
    up2 = Conv2D(256, 3, activation='relu', padding='same')(up2)

    up3 = UpSampling2D(size=(2, 2))(up2)
    up3 = Concatenate()([up3, enc2])
    up3 = Conv2D(128, 3, activation='relu', padding='same')(up3)
    up3 = Conv2D(128, 3, activation='relu', padding='same')(up3)

    up4 = UpSampling2D(size=(2, 2))(up3)
    up4 = Concatenate()([up4, enc1])
    up4 = Conv2D(64, 3, activation='relu', padding='same')(up4)
    up4 = Conv2D(64, 3, activation='relu', padding='same')(up4)

    outputs = Conv2D(num_classes, 1, activation='sigmoid')(up4)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Custom data generator for image-mask pairs
def paired_generator(images_dir, masks_dir, datagen, target_size=(224, 224), batch_size=32):
    image_generator = datagen.flow_from_directory(
        images_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True
    )
    mask_generator = datagen.flow_from_directory(
        masks_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        color_mode='grayscale'
    )
    while True:
        images = next(image_generator)
        masks = next(mask_generator)
        yield (images, masks)

train_generator = paired_generator(TRAIN_DIR, TRAIN_MASK_DIR, train_datagen, batch_size=BATCH_SIZE)
validation_generator = paired_generator(VALID_DIR, VALID_MASK_DIR, valid_datagen, batch_size=BATCH_SIZE)

# Define loss functions
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

# Build and compile model
model = build_simclr_unet(encoder_weights_path='/Users/nadiajelani/projects/wound-segmentation/models/pretrained_base_model.keras')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
)

# Callbacks
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train model
logger.info("Starting U-Net training...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(os.listdir(TRAIN_DIR)) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(os.listdir(VALID_DIR)) // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save final model
model.save(MODEL_SAVE_PATH)
logger.info(f"Final model saved to {MODEL_SAVE_PATH}")

# Visualize predictions
def visualize_predictions(model, images_dir, masks_dir, num_samples=3):
    datagen = ImageDataGenerator(rescale=1./255)
    image_gen = datagen.flow_from_directory(
        images_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=num_samples,
        class_mode=None,
        shuffle=False
    )
    mask_gen = datagen.flow_from_directory(
        masks_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=num_samples,
        class_mode=None,
        shuffle=False,
        color_mode='grayscale'
    )
    images = next(image_gen)
    masks = next(mask_gen)
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(np.uint8)

    for i in range(num_samples):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i, :, :, 0], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

logger.info("Visualizing sample predictions...")
visualize_predictions(model, VALID_DIR, VALID_MASK_DIR)