import warnings
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.cnn_models import CNN1
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical


def plot_image_grid(X, y, nrows, ncols):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for i in range(nrows * ncols):
        # axes[row, col] = plt.imshow(X_train[count])
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(X[i])
        plt.xlabel(np.argmax(y[i]))
    plt.show()


def preprocess_image_input(X):
    X = X.reshape(X.shape[0], 28, 28, 1)
    X = X.astype('float32')

    # RGB values are usually stored as integers to save memory. But doing math on colors is usually done in float because it's easier, more powerful, and more precise. The act of converting floats to integers is called "Quantization", and it throws away precision.

    # Typically, RGB values are encoded as 8-bit integers, which range from 0 to 255. It's an industry standard to think of 0.0f as black and 1.0f as white (max brightness). To convert [0, 255] to [0.0f, 1.0f] all you have to do is divide by 255.0f.
    X /= 255


def main():
    df_train = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/train.csv"))
    df_test = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/test.csv"))

    X, y = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    X_test = df_test.iloc[0:, 0:].values

    preprocess_image_input(X)
    preprocess_image_input(X_test)

    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
    y = y.astype('int32')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42)

    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)

    print(X_train[0].shape)
    # print(X[0:5], y[0:5])
    print(X.shape, y.shape)
    print(type(X), type(y))

    nrows = 2
    ncols = 3

    warnings.simplefilter(action='ignore', category=FutureWarning)
    # plot_image_grid(X_train, y_train, nrows, ncols)

    model = CNN1()
    model.train()


if __name__ == '__main__':
    main()
