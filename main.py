from Models.Unet import *
from Models.vgg16 import *
from Directory import *
from keras import optimizers
from Data import *
import os


MAIN_PATH = "/Users/toniginard/Desktop/TFG"
EXP_NAME = "/Prova"
EXP_PATH = MAIN_PATH + "/Experiments/VGG16" + EXP_NAME


EPOCHS = 100
BATCH_SIZE = 10

NUM_TRAIN_IMAGES = 10
NUM_VAL_IMAGES = 10
NUM_TEST_IMAGES = 5


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = get_model()  # unet()

    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['mse'])  # 'mse', 'mae', 'mape', 'cosine'

    # model.summary()

    # preparar directoris
    create_exp_directory()

    # save_summary(model, EXP)

    # LOAD TRAIN DATA
    l_train = load_data("/Images/Train/left/*.jpg", NUM_TRAIN_IMAGES)  # MAIN_PATH
    r_train = load_data("/Images/Train/right/*.jpg", NUM_TRAIN_IMAGES)  # MAIN_PATH
    d_train = load_data("/Images/Train/depth/*.jpg", NUM_TRAIN_IMAGES)  # MAIN_PATH

    # LOAD VALIDATION DATA
    l_val = load_data("/Images/Validation/left/*.jpg", NUM_VAL_IMAGES)  # MAIN_PATH
    r_val = load_data("/Images/Validation/right/*.jpg", NUM_VAL_IMAGES)  # MAIN_PATH
    d_val = load_data("/Images/Validation/depth/*.jpg", NUM_VAL_IMAGES)  # MAIN_PATH

    # FIT
    history = model.fit(([l_train, r_train]), d_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=([l_val, r_val], d_val))

    save_validation(history.history, EXP_PATH + "/Validation/loss.png")

    # model.save('stereo_to_depth.h5')

    # LOAD PREDICTION DATA
    l_pred = load_data("/Images/Test/left/*.jpg", NUM_TEST_IMAGES)  # MAIN_PATH
    r_pred = load_data("/Images/Test/right/*.jpg", NUM_TEST_IMAGES)  # MAIN_PATH
    pred = model.predict([l_pred, r_pred])
    pred = pred[:, :, :, 0]

    save_predictions(EXP_PATH + "/Predictions", NUM_TEST_IMAGES, pred)


if __name__ == "__main__":
    main()


#  np.tile
