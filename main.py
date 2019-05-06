from Data  import *
from keras import optimizers


def main():
    model = unet()

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    fit(model)


if __name__ == "__main__":
    main()
