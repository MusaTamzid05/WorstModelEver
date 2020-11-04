from keras import models
from keras import layers

def init_model():

    model = models.Sequential()
    model.add(layers.Conv2D(3, strides = 1, kernel_size = (3,3), padding = "same", input_shape = (28, 28, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(155952, activation = "relu"))
    model.add(layers.Reshape((228, 228, 3)))

    model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy")
    return model



def main():

    model = init_model()
    model.summary()


if __name__ == "__main__":
    main()
