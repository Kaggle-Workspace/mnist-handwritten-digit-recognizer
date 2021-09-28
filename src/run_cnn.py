import csv
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.cnn_models import CNN1
from sklearn.model_selection import train_test_split


def main():
    df_train = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/train.csv"))
    df_test = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/test.csv"))

    X, y = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    X = X.reshape(X.shape[0], 28, 28)
    X = X.astype('float32')
    # X /= 255
    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
    y = y.astype('int32')

    X_train, X_valid, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    print(X_train[0].shape)
    # print(X[0:5], y[0:5])
    print(X.shape, y.shape)
    print(type(X), type(y))

    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for i in range(nrows * ncols):
        # axes[row, col] = plt.imshow(X_train[count])
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(X_train[i])
        plt.xlabel(y_train[i])
    plt.show()


if __name__ == '__main__':
    main()
