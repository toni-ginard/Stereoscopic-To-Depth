from keras.preprocessing.image import ImageDataGenerator


def load_train_set(train_path):
    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(100, 100),
        batch_size=25,
        class_mode='binary')

    return train_generator


def load_test_set(test_path):
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(100, 100),
        batch_size=25,
        class_mode='binary')

    return validation_generator
