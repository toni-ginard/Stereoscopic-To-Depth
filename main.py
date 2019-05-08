from Model import unet
from Data import *
from keras import optimizers
import os


def main():
    main_path = "/Users/toniginard/Desktop/TFG/Images/TrainSet"
    model = unet()

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])

    os.chdir(main_path)
    left_gen = input_generator('left')

    os.chdir(main_path)
    right_gen = input_generator('right')

    os.chdir(main_path)
    depth_gen = input_generator('depth')

    history = model.fit_generator([left_gen, right_gen],
                                  steps_per_epoch=5,
                                  epochs=30,
                                  validation_data=depth_gen,
                                  validation_steps=5)


if __name__ == "__main__":
    main()


#  skimage-imread: carregar i passar a escala de grisos
#  metriques entre imatges de profunditat
