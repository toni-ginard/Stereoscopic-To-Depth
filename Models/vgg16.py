import random
import skimage

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, Model, optimizers
from keras.datasets import cifar10


def get_model(input_channels=6, ouput_channels=4, input_spatial_size=(256, 256),
              residual_blocks=5):
    """ Generate an image-to-image model.

    Based on "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", by Justin et al.
    In contrast to the original, instead of using initial padding, the residual blocks use padding
    for simplicity.
    """
    if len(input_spatial_size) == 1:
        input_spatial_size = (input_spatial_size, input_spatial_size)
    assert input_spatial_size[0] % 4 == 0 and input_spatial_size[1] % 4 == 0, 'Input size must by multiple of 4'
    # Define layers
    lyrs = [
        layers.Conv2D(filters=32, kernel_size=(9, 9), strides=1, padding='same'),
        layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same'),
        layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same'),
    ] + [ResidualBlock(num_filters=128) for i in range(residual_blocks)] + [
        layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same'),
        layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, padding='same'),
        layers.Conv2D(filters=ouput_channels, kernel_size=(9, 9), strides=1, padding='same')
    ]
    # Define input-output logic. Assume channels last (tensorflow)
    inp = layers.Input(shape=input_spatial_size + (input_channels,))
    out = inp
    for lyr in lyrs:
        out = lyr(out)
    # Return keras model
    model = Model(inputs=inp, outputs=out)
    return model


class ResidualBlock:
    def __init__(self, num_filters=128):
        self._n = num_filters

    def __call__(self, tensor):
        out = layers.Conv2D(self._n, kernel_size=(3, 3), strides=(1, 1), padding='same')(tensor)
        out = layers.BatchNormalization()(out)
        out = layers.Activation('relu')(out)
        out = layers.Conv2D(self._n, kernel_size=(3, 3), strides=(1, 1), padding='same')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Add()([tensor, out])   # Addition with original input tensor
        out = layers.Activation('relu')(out)
        return out
