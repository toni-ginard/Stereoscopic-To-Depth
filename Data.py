from keras.preprocessing.image import ImageDataGenerator
from Model import *


def train_input_generator(path1, path2):
    train_datagen = ImageDataGenerator(rescale=1./255)

    train_gen_left = train_datagen.flow_from_directory(
        path1,
        target_size=(100, 100),
        batch_size=20,
        class_mode='binary')

    train_gen_right = train_datagen.flow_from_directory(
        path2,
        target_size=(100, 100),
        batch_size=20,
        class_mode='binary')

    while True:
        x1i = train_gen_left.next()
        x2i = train_gen_right.next()
        yield [x1i[0], x2i[0]], x2i[1]  # Yield both images and their mutual label


def train_output_generator(test_path):
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(100, 100),
        batch_size=20,
        class_mode='binary')

    return validation_generator


def fit(model=unet()):
    input_generator = train_input_generator("/Users/toniginard/Desktop/TFG/Images/TrainSet/left",
                                            "/Users/toniginard/Desktop/TFG/Images/TrainSet/right")

    output_generator = train_output_generator("/Users/toniginard/Desktop/TFG/Images/TrainSet/depth")

    history = model.fit_generator(
        input_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=output_generator,
        validation_steps=50)

