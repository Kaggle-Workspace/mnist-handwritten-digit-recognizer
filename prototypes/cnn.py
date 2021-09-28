
import os
from models.cnn_models import CustomCNNModel
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd


def main():
    model = CustomCNNModel()
    data = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/train.csv"))
    X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values
    X = X.reshape(X.shape[0], 28, 28, 1)
    X = X.astype('float32')
    X /= 255
    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
    y = y.astype('int32')



if __name__ == '__main__':
    main()
