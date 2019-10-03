from keras.models import *
from keras import layers


def get_model(input_channels=1, ouput_channels=1, in_left=(64, 64), in_right=(64, 64), residual_blocks=5):
    """ Generate an image-to-image model.

    Based on "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", by Justin et al.
    In contrast to the original, instead of using initial padding, the residual blocks use padding
    for simplicity.
    """
    if len(in_left) == 1:
        in_left = (in_left, in_left)
    assert in_left[0] % 4 == 0 and in_left[1] % 4 == 0, 'Input size must by multiple of 4'
    if len(in_right) == 1:
        in_right = (in_right, in_right)
    assert in_right[0] % 4 == 0 and in_right[1] % 4 == 0, 'Input size must by multiple of 4'
    # Define layers
    lyrs = [
        layers.Conv2D(filters=32, kernel_size=(9, 9), activation='relu', strides=1, padding='same',
                      kernel_initializer='he_normal'),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=2, padding='same',
                      kernel_initializer='he_normal'),
        layers.Dropout(0.5),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=2, padding='same',
                      kernel_initializer='he_normal'),
    ] + [ResidualBlock(num_filters=128) for i in range(residual_blocks)] + [
        layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='relu', strides=2, padding='same',
                               kernel_initializer='he_normal'),
        layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='relu', strides=2, padding='same',
                               kernel_initializer='he_normal'),
        layers.Dropout(0.5),
        layers.Conv2D(filters=ouput_channels, kernel_size=(9, 9), activation='relu', strides=1, padding='same',
                      kernel_initializer='he_normal')
    ]
    # Define input-output logic. Assume channels last (tensorflow)
    left = layers.Input(shape=in_left + (input_channels,))
    right = layers.Input(shape=in_right + (input_channels,))
    inp = layers.concatenate([left, right], axis=-1)
    out = inp
    for lyr in lyrs:
        out = lyr(out)
    # Return keras model
    model = Model(inputs=[left, right], outputs=out)
    return model


class ResidualBlock:
    def __init__(self, num_filters=128):
        self._n = num_filters

    def __call__(self, tensor):
        out = layers.Conv2D(self._n, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                            kernel_initializer='he_normal')(tensor)
        out = layers.BatchNormalization()(out)
        out = layers.Activation('relu')(out)
        out = layers.Conv2D(self._n, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                            kernel_initializer='he_normal')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Add()([tensor, out])   # Addition with original input tensor
        out = layers.Activation('relu')(out)
        return out
