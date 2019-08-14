from Models.n2 import *
from Data import *
import os


def train(exp_name, epochs, batch_size, in_size, n_train_img, n_val_img, n_test_img, pretrained_weights, opt):
    main_path = "/Users/toniginard/TFG/"
    exp_path = main_path + "Entrenaments/" + "n2/" + exp_name + "/"
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    model = get_model(in_left=(in_size, in_size), in_right=(in_size, in_size))

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    save_summary(model, exp_path)

    train_generator = get_train_generator(main_path + "Images/Train/left",
                                          main_path + "Images/Train/right",
                                          main_path + "Images/Train/depth",
                                          in_size,
                                          batch_size)

    val_generator = get_train_generator(main_path + "Images/Validation/left",
                                        main_path + "Images/Validation/right",
                                        main_path + "Images/Validation/depth",
                                        in_size,
                                        batch_size)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=n_train_img / batch_size,
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  validation_steps=n_val_img / batch_size)

    save_validation(history.history, exp_path + "/loss.png")

    test_generator = get_test_generator(main_path + "Images/Test/left",
                                        main_path + "Images/Test/right",
                                        in_size,
                                        1)

    predictions = model.predict_generator(test_generator, steps=n_test_img)
    predictions = predictions[:, :, :, 0]
    save_predictions(exp_path + "Predictions", n_test_img, predictions)
