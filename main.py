from Models.Unet import *
from Models.n2 import *
from keras import optimizers
from Data import *
import os


MAIN_PATH = "/Users/toniginard/Desktop/TFG/"
EXP_NAME = "n2/Prova/"
EXP_PATH = MAIN_PATH + "Entrenaments/" + EXP_NAME


EPOCHS = 2
BATCH_SIZE = 2

NUM_TRAIN_IMAGES = 10
NUM_VAL_IMAGES = 6
NUM_TEST_IMAGES = 2

IN_SIZE = 256


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = get_model(in_left=(IN_SIZE, IN_SIZE), in_right=(IN_SIZE, IN_SIZE))  # unet()

    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['mse'])

    # model.summary()

    save_summary(model, EXP_PATH)

    train_generator = get_train_generator(MAIN_PATH + "Images/Train/left",
                                          MAIN_PATH + "Images/Train/right",
                                          MAIN_PATH + "Images/Train/depth",
                                          IN_SIZE,
                                          BATCH_SIZE)

    val_generator = get_train_generator(MAIN_PATH + "Images/Validation/left",
                                        MAIN_PATH + "Images/Validation/right",
                                        MAIN_PATH + "Images/Validation/depth",
                                        IN_SIZE,
                                        BATCH_SIZE)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=NUM_TRAIN_IMAGES / BATCH_SIZE,
                                  epochs=EPOCHS,
                                  validation_data=val_generator,
                                  validation_steps=NUM_VAL_IMAGES / BATCH_SIZE)

    save_validation(history.history, EXP_PATH + "/loss.png")

    test_generator = get_test_generator(MAIN_PATH + "Images/Test/left",
                                        MAIN_PATH + "Images/Test/right",
                                        IN_SIZE,
                                        BATCH_SIZE)

    predictions = model.predict_generator(test_generator,
                                          steps=NUM_TEST_IMAGES)

    # save_predictions(EXP_PATH + "Predictions", NUM_TEST_IMAGES, predictions)


if __name__ == "__main__":
    main()


#  np.tile
