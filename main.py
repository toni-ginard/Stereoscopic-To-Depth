from Train import *
from keras import optimizers


def main():

    rmsprop = optimizers.RMSprop(lr=1e-4)

    train(exp_name="Prova",
          epochs=2,
          batch_size=2,
          in_size=128,
          n_train_img=4,
          n_val_img=2,
          n_test_img=2,
          pretrained_weights=None,
          opt=rmsprop)


if __name__ == "__main__":
    main()
