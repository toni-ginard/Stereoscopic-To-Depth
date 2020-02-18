#!/usr/bin/env python
# -*- coding: utf-8 -*-


from Models.n2 import get_architecture
from Data import *
from Constants import *
from keras.callbacks import EarlyStopping
from keras import optimizers
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


def training(pretrained_weights):
    """ Load a model, compile, train and make predictions.

    :param pretrained_weights: file containing pretrained weights.
    :return: history object, corresponding to the training.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # GET MODEL AND COMPILE
    model = get_architecture(in_left=(IMG_SIZE, IMG_SIZE), in_right=(IMG_SIZE, IMG_SIZE))
    model.compile(loss=custom_mse, optimizer=optimizers.RMSprop(lr=1e-4), metrics=[custom_mse])

    if pretrained_weights:
        model.load_weights(EXP_NAME + pretrained_weights)

    save_summary(model, EXP_NAME)

    # GET DATA
    train_generator = get_train_generator(
        DATA_PATH + "/Train/left",
        DATA_PATH + "/Train/right",
        DATA_PATH + "/Train/depth"
    )

    val_generator = get_train_generator(
        DATA_PATH + "/Validation/left",
        DATA_PATH + "/Validation/right",
        DATA_PATH + "/Validation/depth"
    )

    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=5
    )

    # TRAIN
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=N_TRAIN_IMG / BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=N_VAL_IMG / BATCH_SIZE,
        callbacks=[es]
    )

    model.save_weights(EXP_NAME + "/weights.h5")
    save_validation(history, EXP_NAME + "/loss.png")

    # PREDICTIONS
    test_generator = get_test_generator(
        DATA_PATH + "/Test/left",
        DATA_PATH + "/Test/right"
    )

    predictions = model.predict_generator(test_generator, steps=N_TEST_IMG)
    predictions = predictions[:, :, :, 0]
    save_predictions(predictions, EXP_NAME + "/Predictions")

