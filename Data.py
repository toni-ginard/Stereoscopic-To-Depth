from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from skimage import color
from numpy import *
import matplotlib.pyplot as plt


# def train_input_generator(gen, path1, path2):
#     train_gen_left = gen.flow_from_directory(path1, target_size=(64, 64), batch_size=20)
#
#     train_gen_right = gen.flow_from_directory(path2, target_size=(64, 64), batch_size=20)
#
#     while True:
#         X1i = train_gen_left.next()
#         X2i = train_gen_right.next()
#         yield [X1i[0], X2i[0]], X1i[1]


# def input_generator(path):
#     train_datagen = ImageDataGenerator(rescale=1./255)
#     input_gen = train_datagen.flow_from_directory(path, target_size=(64, 64), batch_size=20, color_mode="grayscale")
#     return input_gen


# def train_output_generator(gen, path):
#     validation_generator = gen.flow_from_directory(path, target_size=(64, 64), batch_size=20)
#     return validation_generator


def load_data(path):
    images = io.imread_collection(path)
    list = []
    """for i in images:
        list.append(color.rgb2gray(i))"""
    for i in range(len(images)):
        list.append(resize(color.rgb2gray(images[i]), [64, 64]))
        list[i] = list[i][:, :, newaxis]
    return array(list)


def validation(history):
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
