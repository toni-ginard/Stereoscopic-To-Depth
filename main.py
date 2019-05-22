from Model import unet
from Data import *
from keras import optimizers
import os
from PIL import Image


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = unet()

    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['mse', 'mae', 'mape', 'cosine'])

    # Training data
    l_train = load_data("/Users/toniginard/Desktop/TFG/Images/Train/left/*.jpg")
    r_train = load_data("/Users/toniginard/Desktop/TFG/Images/Train/right/*.jpg")
    d_train = load_data("/Users/toniginard/Desktop/TFG/Images/Train/depth/*.jpg")

    # Validation data
    l_val = load_data("/Users/toniginard/Desktop/TFG/Images/Validation/left/*.jpg")
    r_val = load_data("/Users/toniginard/Desktop/TFG/Images/Validation/right/*.jpg")
    d_val = load_data("/Users/toniginard/Desktop/TFG/Images/Validation/depth/*.jpg")

    history = model.fit(([l_train, r_train]), d_train, epochs=10, batch_size=2, validation_data=([l_val, r_val], d_val))

    # validation(history.history)

    # model.save('stereo_to_depth.h5')

    # Prediction data
    l_pred = load_data("/Users/toniginard/Desktop/TFG/Images/Proves/left/*.jpg")
    r_pred = load_data("/Users/toniginard/Desktop/TFG/Images/Proves/right/*.jpg")
    pred = model.predict([l_pred, r_pred])
    pred = pred[:, :, :, 0]

    print("IM: ", pred[0])
    print("SHAPE: ", pred[0].shape)
    plt.imsave("/Users/toniginard/Desktop/im.jpg", pred[0])


if __name__ == "__main__":
    main()


#  metriques entre imatges de profunditat
#  visualitzar dades, conjunts de validació i entrenament, representar imatges resultants (què fa exactament)
