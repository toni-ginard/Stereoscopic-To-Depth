from Models.Unet import *
from Data import *
from keras import optimizers
import os


MAIN_PATH = "/Users/toniginard/Desktop/TFG"
EXP = MAIN_PATH + "/Experiments/Prova"

EPOCHS = 10
BATCH_SIZE = 2

NUM_TRAIN_IMAGES = 10
NUM_VAL_IMAGES = 1
NUM_TEST_IMAGES = 2


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = unet()

    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['mse'])  # 'mse', 'mae', 'mape', 'cosine'

    save_summary(model, EXP)

    # LOAD TRAIN DATA
    l_train = load_data(MAIN_PATH + "/Images/Train/left/*.jpg", NUM_TRAIN_IMAGES)
    r_train = load_data(MAIN_PATH + "/Images/Train/right/*.jpg", NUM_TRAIN_IMAGES)
    d_train = load_data(MAIN_PATH + "/Images/Train/depth/*.jpg", NUM_TRAIN_IMAGES)

    # LOAD VALIDATION DATA
    l_val = load_data(MAIN_PATH + "/Images/Validation/left/*.jpg", NUM_VAL_IMAGES)
    r_val = load_data(MAIN_PATH + "/Images/Validation/right/*.jpg", NUM_VAL_IMAGES)
    d_val = load_data(MAIN_PATH + "/Images/Validation/depth/*.jpg", NUM_VAL_IMAGES)

    # FIT
    history = model.fit(([l_train, r_train]), d_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=([l_val, r_val], d_val))

    save_validation(history.history, EXP + "/Validation/loss.png")

    # model.save('stereo_to_depth.h5')

    # LOAD PREDICTION DATA
    l_pred = load_data(MAIN_PATH + "/Images/Test/left/*.jpg", NUM_TEST_IMAGES)
    r_pred = load_data(MAIN_PATH + "/Images/Test/right/*.jpg", NUM_TEST_IMAGES)
    pred = model.predict([l_pred, r_pred])
    pred = pred[:, :, :, 0]

    save_predictions(EXP + "/Predictions", NUM_TEST_IMAGES, pred)


if __name__ == "__main__":
    main()


#  np.tile
