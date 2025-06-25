from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def build_simclr_unet(input_shape=(224, 224, 3), num_classes=1, weights=None):
    inputs = layers.Input(shape=input_shape)

    # Encoder (ResNet50) with optional weights (e.g., 'imagenet')
    base_model = ResNet50(include_top=False, weights=weights, input_tensor=inputs)

    # Skip connections
    skip1 = base_model.get_layer("conv1_relu").output
    skip2 = base_model.get_layer("conv2_block3_out").output
    skip3 = base_model.get_layer("conv3_block4_out").output
    skip4 = base_model.get_layer("conv4_block6_out").output
    bridge = base_model.get_layer("conv5_block3_out").output

    # Decoder
    x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(bridge)
    x = layers.concatenate([x, skip4])
    x = layers.Conv2D(512, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(512, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.concatenate([x, skip3])
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.concatenate([x, skip2])
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.concatenate([x, skip1])
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)

    output_activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, 1, activation=output_activation)(x)

    return models.Model(inputs=inputs, outputs=outputs)
