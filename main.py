from Model import unet
from Data import *
from keras import optimizers
import os


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = unet()

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])

    left_img = load_data("/Users/toniginard/Desktop/TFG/Images/TrainSet/left/*.jpg")
    right_img = load_data("/Users/toniginard/Desktop/TFG/Images/TrainSet/right/*.jpg")
    depth_img = load_data("/Users/toniginard/Desktop/TFG/Images/TrainSet/depth/*.jpg")

    history = model.fit(([left_img, right_img]), depth_img, epochs=2, batch_size=2)

    model.save('stereo_to_depth.h5')


if __name__ == "__main__":
    main()


#  metriques entre imatges de profunditat
