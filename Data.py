from keras.preprocessing.image import ImageDataGenerator
import os


def train_input_generator(gen, path1, path2):
    train_gen_left = gen.flow_from_directory(path1, target_size=(64, 64), batch_size=20)

    train_gen_right = gen.flow_from_directory(path2, target_size=(64, 64), batch_size=20)

    while True:
        X1i = train_gen_left.next()
        X2i = train_gen_right.next()
        yield [X1i[0], X2i[0]], X1i[1]


def input_generator(path):
    train_datagen = ImageDataGenerator(rescale=1./255)
    input_gen = train_datagen.flow_from_directory(path, target_size=(64, 64), batch_size=20, color_mode="grayscale")
    return input_gen


def train_output_generator(gen, path):
    validation_generator = gen.flow_from_directory(path, target_size=(64, 64), batch_size=20)
    return validation_generator
