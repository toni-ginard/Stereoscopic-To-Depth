from Models.Unet import *
from Models.n2 import *
from keras import optimizers
from Data import *
import os


MAIN_PATH = "/Users/toniginard/Desktop/TFG/"
EXP_NAME = "n2/Prova/"
EXP_PATH = MAIN_PATH + "Entrenaments/" + EXP_NAME


EPOCHS = 10
BATCH_SIZE = 2

NUM_TRAIN_IMAGES = 10
NUM_VAL_IMAGES = 6
NUM_TEST_IMAGES = 1

IN_SIZE = 256


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = get_model(in_left=(IN_SIZE, IN_SIZE), in_right=(IN_SIZE, IN_SIZE))  # unet()

    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['mse'])

    # model.summary()

    save_summary(model, EXP_PATH)

    # LOAD TRAIN IMAGES
    l_train_gen = generator(MAIN_PATH + "Images/Train/left", IN_SIZE, BATCH_SIZE)
    r_train_gen = generator(MAIN_PATH + "Images/Train/right", IN_SIZE, BATCH_SIZE)
    d_train_gen = generator(MAIN_PATH + "Images/Train/depth", IN_SIZE, BATCH_SIZE)

    # LOAD VALIDATION IMAGES
    l_val_gen = generator(MAIN_PATH + "Images/Validation/left", IN_SIZE, BATCH_SIZE)
    r_val_gen = generator(MAIN_PATH + "Images/Validation/right", IN_SIZE, BATCH_SIZE)
    d_val_gen = generator(MAIN_PATH + "Images/Validation/depth", IN_SIZE, BATCH_SIZE)

    train_generator = multiple_generator(l_train_gen, r_train_gen, d_train_gen)
    val_generator = multiple_generator(l_val_gen, r_val_gen, d_val_gen)

    history = model.fit_generator(train_generator,
                                  epochs=EPOCHS,
                                  steps_per_epoch=NUM_TRAIN_IMAGES/BATCH_SIZE,
                                  validation_data=val_generator,
                                  validation_steps=NUM_VAL_IMAGES/BATCH_SIZE,
                                  shuffle=False)

    save_validation(history.history, EXP_PATH + "/loss.png")

    # LOAD PREDICTION DATA
    l_pred = generator(MAIN_PATH + "Images/Test/left", IN_SIZE, BATCH_SIZE)
    r_pred = generator(MAIN_PATH + "Images/Test/right", IN_SIZE, BATCH_SIZE)

    # LOAD PREDICTION DATA
#   l_pred = load_data(MAIN_PATH + "Images/Test/left/*.jpg", NUM_TEST_IMAGES, IN_SIZE)  # MAIN_PATH
#   r_pred = load_data(MAIN_PATH + "Images/Test/right/*.jpg", NUM_TEST_IMAGES, IN_SIZE)  # MAIN_PATH
#   pred = model.predict([l_pred, r_pred])
#   pred = pred[:, :, :, 0]

#   save_predictions(EXP_PATH + "Predictions", NUM_TEST_IMAGES, pred)


if __name__ == "__main__":
    main()


#  np.tile
