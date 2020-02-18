from Models.n2 import *
from Data import *
from keras import optimizers

LEFT_PATH = "Test_256/left"
RIGHT_PATH = "Test_256/right"
IMG_SIZE = 256
N_TEST_IMG = 1
WEIGHTS = "weights_FINAL.h5"


model = get_architecture(in_left=(IMG_SIZE, IMG_SIZE), in_right=(IMG_SIZE, IMG_SIZE))
model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['mse'])
model.load_weights(WEIGHTS)

test_generator = get_test_generator(LEFT_PATH,
                                    RIGHT_PATH)

pred = model.predict_generator(test_generator, steps=N_TEST_IMG)
pred = pred[:, :, :, 0]
save_predictions(pred, "Predictions")
