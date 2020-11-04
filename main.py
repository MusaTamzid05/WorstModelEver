import tensorflow as tf
import cv2

def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        print(e)

limit_gpu()

from keras import models
from keras import layers
from debugger.trainer import Trainer
from debugger.predictor import Predictor
from data_loader import DataLoader


def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        print(e)

def init_model():

    model = models.Sequential()
    model.add(layers.Conv2DTranspose(18, strides = 1, kernel_size = (3,3), padding = "same", input_shape = (28, 28, 3)))
    model.add(layers.Conv2DTranspose(9, strides = 2, kernel_size = (3,3), padding = "same"))
    model.add(layers.Conv2DTranspose(6, strides = 2, kernel_size = (3,3), padding = "same"))
    model.add(layers.Conv2DTranspose(3, strides = 2, kernel_size = (3,3), padding = "same"))


    model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model



def main():

    model = init_model()
    model.summary()

    data_loader = DataLoader(x_dir_path = "28x28", y_dir_path = "224x224")
    x_train , y_train, x_val, y_val = data_loader.run()

    trainer = Trainer(
            model = model,
            x_train = x_train,
            y_train = y_train,
            x_val = x_val,
            y_val = y_val
            )

    trainer.train(epochs = 1000, batch_size = 4, save_dir = "results")


def test():

    data_loader = DataLoader(x_dir_path = "28x28", y_dir_path = "224x224")
    x_train , y_train, x_val, y_val = data_loader.run()

    predictor = Predictor(model_path = "results/model.h5")
    result = predictor.predict(x_val[0])

    cv2.imwrite("result.png", result)

if __name__ == "__main__":
    test()
