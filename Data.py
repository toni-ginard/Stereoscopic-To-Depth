from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from skimage import color
from numpy import *
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import os


def get_train_generator(l_path, r_path, d_path, in_size, batch_size):
    l_datagen = ImageDataGenerator(rescale=1./255)
    r_datagen = ImageDataGenerator(rescale=1./255)
    d_datagen = ImageDataGenerator(rescale=1./255)

    l_generator = l_datagen.flow_from_directory(l_path,
                                                class_mode=None,
                                                color_mode='grayscale',
                                                target_size=(in_size, in_size),
                                                batch_size=batch_size,
                                                shuffle=False)

    r_generator = r_datagen.flow_from_directory(r_path,
                                                class_mode=None,
                                                color_mode='grayscale',
                                                target_size=(in_size, in_size),
                                                batch_size=batch_size,
                                                shuffle=False)

    d_generator = d_datagen.flow_from_directory(d_path,
                                                class_mode=None,
                                                color_mode='grayscale',
                                                target_size=(in_size, in_size),
                                                batch_size=batch_size,
                                                shuffle=False)

    in_generator = zip(l_generator, r_generator)
    train_generator = zip(in_generator, d_generator)
    for (in_img, d_img) in train_generator:
        yield([in_img[0], in_img[1]], d_img)


def get_test_generator(l_path, r_path, in_size, batch_size):
    l_datagen = ImageDataGenerator(rescale=1./255)
    r_datagen = ImageDataGenerator(rescale=1./255)

    l_generator = l_datagen.flow_from_directory(l_path,
                                                class_mode=None,
                                                color_mode='grayscale',
                                                target_size=(in_size, in_size),
                                                batch_size=batch_size,
                                                shuffle=False)

    r_generator = r_datagen.flow_from_directory(r_path,
                                                class_mode=None,
                                                color_mode='grayscale',
                                                target_size=(in_size, in_size),
                                                batch_size=batch_size,
                                                shuffle=False)
    test_generator = zip(l_generator, r_generator)
    for (l_img, r_img) in test_generator:
        yield(l_img, r_img)


def load_data(path, num_images, in_size):
    images = io.imread_collection(path)
    list = []
    """for i in images:
        list.append(color.rgb2gray(i))"""
    for i in range(num_images):
        list.append(resize(color.rgb2gray(images[i]), [in_size, in_size]))
        list[i] = list[i][:, :, newaxis]
    my_array = array(list)
    my_array = my_array[:num_images]
    return my_array  # array(list)  # [:num_images]


def save_validation(history, path):
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path)


def save_summary(model, path):
    open(os.path.join(path, 'summary.txt'), 'w')
    with open(path + '/' + 'summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()


def save_predictions(path, num_images, predictions):
    # plt.imsave(path + "pred0.png", predictions, cmap='gray')
    for i in range(num_images):
        name = path + "/pred" + str(i) + ".png"
        plt.imsave(name, predictions[i], cmap='gray')
