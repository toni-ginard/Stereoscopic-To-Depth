from Train import *
from keras import optimizers
import time


def main():
    exp = "MT4"
    ep = 2
    bs = 2
    train_img = 8
    val_img = 4
    rmsprop = optimizers.RMSprop(lr=1e-4)

    # img_size, n_test_img, conj, pretrained_weights
    exps_params = [(64,  1,  '1', None),
                   (128, 1,  '1', "weights_" + exp + ".h5"),
                   (256, 1,  '1', "weights_" + exp + ".h5"),
                   (64,  1,  '2', "weights_" + exp + ".h5"),
                   (128, 1,  '2', "weights_" + exp + ".h5"),
                   (256, 1,  '2', "weights_" + exp + ".h5"),
                   (64,  1,  '3', "weights_" + exp + ".h5"),
                   (128, 1,  '3', "weights_" + exp + ".h5"),
                   (256, 10, '3', "weights_" + exp + ".h5")]

    histories = []

    start_time = time.time()

    for i in exps_params:
        history = train(exp_name=exp,
                        epochs=ep,
                        batch_size=bs,
                        img_size=i[0],
                        n_train_img=train_img,
                        n_val_img=val_img,
                        n_test_img=i[1],
                        conj=i[2],
                        pretrained_weights=i[3],
                        opt=rmsprop)
        histories.append(history)

    save_validation(histories, "Entrenaments/" + exp + "/loss.png")

    print("- %s seconds -" % (time.time() - start_time))


if __name__ == "__main__":
    main()
