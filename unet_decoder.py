import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Input, concatenate,
    BatchNormalization, Activation, UpSampling2D
)
from tensorflow.keras.applications import ResNet50

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def decoder_block(input_tensor, skip_tensor, num_filters):
    x = UpSampling2D((2, 2))(input_tensor)
    x = concatenate([x, skip_tensor])
    x = conv_block(x, num_filters)
    return x

def build_simclr_unet(input_shape=(224, 224, 3), num_classes=1, encoder_weights_path=None):
    inputs = Input(shape=input_shape)

    # Load ResNet50 Encoder
    base_model = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    if encoder_weights_path:
        base_model.load_weights(encoder_weights_path)

    # Extract Encoder Features
    skip1 = base_model.get_layer("conv1_relu").output        # 112x112
    skip2 = base_model.get_layer("conv2_block3_out").output  # 56x56
    skip3 = base_model.get_layer("conv3_block4_out").output  # 28x28
    skip4 = base_model.get_layer("conv4_block6_out").output  # 14x14
    bottleneck = base_model.get_layer("conv5_block3_out").output  # 7x7

    # Decoder
    d1 = decoder_block(bottleneck, skip4, 512)
    d2 = decoder_block(d1, skip3, 256)
    d3 = decoder_block(d2, skip2, 128)
    d4 = decoder_block(d3, skip1, 64)

    # Add final upsampling to match input resolution
    x = UpSampling2D((2, 2))(d4)  # 112 -> 224
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model
