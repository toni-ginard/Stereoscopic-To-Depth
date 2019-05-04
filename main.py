from Model import *
from keras import optimizers


def main():
    model = unet()

    # model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    # model.fit()


if __name__ == "__main__":
    main()
