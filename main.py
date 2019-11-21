#!/usr/bin/env python
# -*- coding: utf-8 -*-


from Train import train
from Data import save_validation
from Constants import *
from keras import optimizers
import time


def main():

    optimizer = optimizers.RMSprop(lr=1e-4)

    start_time = time.time()

    history = train(exp_name=EXP,
                    folder='1',
                    pretrained_weights=None,
                    optimizer=optimizer)

    save_validation(history, "Entrenaments/" + EXP + "/loss.png")

    print("> %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
