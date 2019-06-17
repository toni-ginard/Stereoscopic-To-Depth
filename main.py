from Model import *
from Data import *
from keras import optimizers
import os


num_train_images = 5
num_val_images = 1
exp = "/Users/toniginard/Desktop/TFG/Experiments/Prova"
main_path = "/Users/toniginard/Desktop/TFG/"

epochs = 5
batch_size = 2


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = unet()

    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['mse'])  # 'mse', 'mae', 'mape', 'cosine'

    save_model(model, exp)

    # Load training data
    l_train = load_data(main_path + "Images/Train/left/*.jpg")
    r_train = load_data(main_path + "Images/Train/right/*.jpg")
    d_train = load_data(main_path + "Images/Train/depth/*.jpg")
    l_train = l_train[:num_train_images]
    r_train = r_train[:num_train_images]
    d_train = d_train[:num_train_images]

    # Load validation data
    l_val = load_data(main_path + "Images/Validation/left/*.jpg")
    r_val = load_data(main_path + "Images/Validation/right/*.jpg")
    d_val = load_data(main_path + "Images/Validation/depth/*.jpg")
    l_val = l_val[:num_train_images]
    r_val = r_val[:num_train_images]
    d_val = d_val[:num_train_images]

    # Entrenar
    history = model.fit(([l_train, r_train]), d_train, epochs=epochs, batch_size=batch_size,
                        validation_data=([l_val, r_val], d_val))

    save_validation(history.history, exp + "/Validation/loss.png")

    # model.save('stereo_to_depth.h5')

    # Load prediction data
    l_pred = load_data(main_path + "Images/Test/left/*.jpg")
    r_pred = load_data(main_path + "Images/Test/right/*.jpg")
    pred = model.predict([l_pred, r_pred])
    pred = pred[0, :, :, 0]

    plt.imsave(exp + "/Predictions/pred0.png", pred, cmap='gray')


if __name__ == "__main__":
    main()


#  np.tile
