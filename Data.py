#!/usr/bin/env python
# -*- coding: utf-8 -*-


from Constants import *
from contextlib import redirect_stdout
from keras.preprocessing.image import ImageDataGenerator
from numpy import *
import matplotlib.pyplot as plt
import os
from collections import Iterable


def adjust_data():
    """ Adjust data for data augmentation.
    :return: ImageDataGenerator object.
    """
    return ImageDataGenerator(rescale=1./255)


def get_train_generator(l_path, r_path, d_path):
    """ Creates a data generator for the training. Creates one generator each for left, right (input) and
    depth (prediction) images. Joins left and right.

    :param l_path: directory for input 'left eye' images.
    :param r_path: directory for input 'right eye' images.
    :param d_path: directory for target depth image.
    :return: train generator.
    """
    l_datagen = adjust_data()
    r_datagen = adjust_data()
    d_datagen = adjust_data()
    l_generator = l_datagen.flow_from_directory(
        l_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    r_generator = r_datagen.flow_from_directory(
        r_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    d_generator = d_datagen.flow_from_directory(
        d_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    in_generator = zip(l_generator, r_generator)
    train_generator = zip(in_generator, d_generator)
    for (in_img, d_img) in train_generator:
        yield([in_img[0], in_img[1]], d_img)


def get_test_generator(l_path, r_path):
    """ Creates a data generator for the testing. Creates one generator each for left and right
    and joins them.

    :param l_path: directory for input 'left eye' images.
    :param r_path: directory for input 'right eye' images.
    :return: train generator.
    """
    l_datagen = adjust_data()
    r_datagen = adjust_data()

    l_generator = l_datagen.flow_from_directory(
        l_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=False
    )

    r_generator = r_datagen.flow_from_directory(
        r_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=False
    )

    test_generator = zip(l_generator, r_generator)
    for (l_img, r_img) in test_generator:
        yield[l_img, r_img]


def save_validation(history, directory):
    """ Saves a graphic for our training and validation loss. Y axis is in logarithmic scale.

    :param history: History object. Its `History.history` attribute is a record of training loss
        and validation loss values at successive epochs.
    :param directory: directory where to store the graphic.
    """
    loss = []
    val_loss = []

    if isinstance(history, Iterable):
        for i in history:
            loss += i.history['loss']
            val_loss += i.history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(directory)


def save_summary(model, directory):
    """ Saves summary of a model in the specified path.

    :param model: net model.
    :param directory: directory where to store the summary.
    """
    open(os.path.join(directory, 'summary.txt'), 'w')
    with open(directory + '/' + 'summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()


def save_predictions(predictions, directory):
    """ Saves predictions made from a model.

    :param predictions: numpy.array
    :param directory: directory where to store the predictions.
    """
    if N_TEST_IMG > 0:
        for i in range(N_TEST_IMG):
            name = directory + "/pred" + str(i) + ".png"
            plt.imsave(name, predictions[i], cmap='gray')
