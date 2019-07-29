from Models.Unet import *
from Models.n2 import *
from keras import optimizers
from Data import *
import os


IN_SIZE = 256


def train(exp_name, epochs, batch_size, n_train_img, n_val_img, n_test_img):
    main_path = "/Users/toniginard/Desktop/TFG/"
    exp_path = main_path + "Entrenaments/" + "n2/" + exp_name + "/"
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = get_model(in_left=(IN_SIZE, IN_SIZE), in_right=(IN_SIZE, IN_SIZE))  # unet()

    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['mse'])

    save_summary(model, exp_path)

    train_generator = get_train_generator(main_path + "Images/Train/left",
                                          main_path + "Images/Train/right",
                                          main_path + "Images/Train/depth",
                                          IN_SIZE,
                                          batch_size)

    val_generator = get_train_generator(main_path + "Images/Validation/left",
                                        main_path + "Images/Validation/right",
                                        main_path + "Images/Validation/depth",
                                        IN_SIZE,
                                        batch_size)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=n_train_img / batch_size,
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  validation_steps=n_val_img / batch_size)

    save_validation(history.history, exp_path + "/loss.png")

    testgenerator = get_test_generator(main_path + "Images/Test/left",
                                       main_path + "Images/Test/right",
                                       IN_SIZE,
                                       1)

    l_test_generator = test_generator(main_path + "Images/Test/left",
                                      IN_SIZE,
                                      1)

    r_test_generator = test_generator(main_path + "Images/Test/right",
                                      IN_SIZE,
                                      1)

    predictions = model.predict_generator(testgenerator, steps=n_test_img)
    predictions = predictions[:, :, :, 0]
    save_predictions(exp_path + "Predictions", n_test_img, predictions)
