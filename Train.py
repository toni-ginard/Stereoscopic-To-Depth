from Models.n2 import *
from keras.callbacks import EarlyStopping
from Data import *
import os
import keras.backend as k


def custom_mse(y_true, y_pred):
    mean_true = k.mean(y_true)
    mean_pred = k.mean(y_pred)
    alpha = mean_true / mean_pred
    return K.mean(K.square(y_true - alpha * y_pred))


def train(exp_name, epochs, batch_size, img_size, n_train_img, n_val_img, n_test_img, conj, pretrained_weights, opt):
    exp_path = "Entrenaments/" + exp_name + "/"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = get_model(in_left=(img_size, img_size), in_right=(img_size, img_size))

    model.compile(loss=custom_mse, optimizer=opt, metrics=[custom_mse])

    if pretrained_weights:
        model.load_weights(exp_path + pretrained_weights)

    save_summary(model, exp_path, exp_name)

    train_generator = get_train_generator("Img" + str(img_size) + "_" + conj + "/Train/left",
                                          "Img" + str(img_size) + "_" + conj + "/Train/right",
                                          "Img" + str(img_size) + "_" + conj + "/Train/depth",
                                          img_size,
                                          batch_size)

    val_generator = get_train_generator("Img" + str(img_size) + "_" + conj + "/Validation/left",
                                        "Img" + str(img_size) + "_" + conj + "/Validation/right",
                                        "Img" + str(img_size) + "_" + conj + "/Validation/depth",
                                        img_size,
                                        batch_size)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=n_train_img / batch_size,
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  validation_steps=n_val_img / batch_size,
                                  callbacks=[es])

    # save_validation(history.history, exp_path + "/loss_" + exp_name + ".png")
    model.save_weights(exp_path + "/weights_" + exp_name + ".h5")

    test_generator = get_test_generator("Img" + str(img_size) + "_" + conj + "/Test/left",
                                        "Img" + str(img_size) + "_" + conj + "/Test/right",
                                        img_size,
                                        1)

    pred = model.predict_generator(test_generator, steps=n_test_img)
    pred = pred[:, :, :, 0]
    save_predictions(exp_path + "Predictions_" + exp_name, n_test_img, pred)

    return history
