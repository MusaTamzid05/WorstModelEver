import os
import cv2
import numpy as np

class DataLoader:

    def __init__(self, x_dir_path , y_dir_path):
        self._load_data(x_dir_path, y_dir_path)


    def _load_data(self, x_dir_path, y_dir_path):
        image_file_names = sorted(os.listdir(x_dir_path))

        self.x_images = []
        self.y_images = []


        for filename in image_file_names:
            x_path = os.path.join(x_dir_path, filename)
            y_path = os.path.join(y_dir_path, filename)

            x_image = cv2.imread(x_path)

            if x_image is None:
                raise RuntimeError(f"{x_path} load error")

            y_image = cv2.imread(y_path)


            if y_image is None:
                raise RuntimeError(f"{y_path} load error")

            self.x_images.append(x_image)
            self.y_images.append(y_image)


    def run(self, train_size = 0.8):
        train_index = int(len(self.x_images) * train_size)

        train_x = self.x_images[:train_index]
        train_y = self.y_images[:train_index]


        val_x = self.x_images[train_index:]
        val_y = self.y_images[train_index:]


        return np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y)




def main():
    data_loader = DataLoader(x_dir_path = "28x28", y_dir_path = "224x224")
    train_x, train_y, val_x, val_y = data_loader.run()

    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(val_y.shape)


if __name__ == "__main__":
    main()

