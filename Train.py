#!/usr/bin/env python
# -*- coding: utf-8 -*-


from Models.n2 import get_model
from Data import *
from Constants import *
from keras.callbacks import EarlyStopping
import keras.backend as k


def custom_mse(y_true, y_pred):
    mean_true = k.mean(y_true)
    mean_pred = k.mean(y_pred)
    alpha = mean_true / mean_pred
    return k.mean(k.square(y_true - alpha * y_pred))


def norm_mse(y_true, y_pred):
    mean_true = k.mean(y_true)
    mean_pred = k.mean(y_pred)
    return k.mean(k.square(y_true / mean_true - y_pred / mean_pred))


def train(exp_name, folder, pretrained_weights, optimizer):
    """ Load a model, compile, train it and make some predictions.

    :param exp_name: folder where the results of the training will be stored.
    :param folder: specific images' subset.
    :param pretrained_weights: file containing pretrained weights.
    :param optimizer:
    :return:
    """
    exp_path = "Entrenaments/" + exp_name + "/"
    data_path = "Img" + str(IMG_SIZE) + "_" + folder

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = get_model(in_left=(IMG_SIZE, IMG_SIZE), in_right=(IMG_SIZE, IMG_SIZE))

    model.compile(loss=custom_mse, optimizer=optimizer, metrics=[custom_mse])

    if pretrained_weights:
        model.load_weights(exp_path + pretrained_weights)

    save_summary(model, exp_path)

    train_generator = get_train_generator(data_path + "/Train/left",
                                          data_path + "/Train/right",
                                          data_path + "/Train/depth")

    val_generator = get_train_generator(data_path + "/Validation/left",
                                        data_path + "/Validation/right",
                                        data_path + "/Validation/depth")

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=N_TRAIN_IMG / BATCH_SIZE,
                                  epochs=EPOCHS,
                                  validation_data=val_generator,
                                  validation_steps=N_VAL_IMG / BATCH_SIZE,
                                  callbacks=[es])

    model.save_weights(exp_path + "/weights.h5")

    test_generator = get_test_generator(data_path + "/Test/left",
                                        data_path + "/Test/right")

    predictions = model.predict_generator(test_generator, steps=N_TEST_IMG)
    predictions = predictions[:, :, :, 0]
    save_predictions(predictions, exp_path + "Predictions_" + exp_name)

    return history
