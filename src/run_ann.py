
import os

from scipy.sparse.construct import rand
from tensorflow.python.keras.backend import shape
from models.ann_models import ANN1, ANN2
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/train.csv"))
    X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values
    print("The shape of the input is: ", X.shape, y.shape)
    X_train, X_valid, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # model = ANN1()
    model = ANN2()
    model.train(X_train, X_valid, y_train, y_test, epochs=3, batch_size=32)


if __name__ == '__main__':
    main()
