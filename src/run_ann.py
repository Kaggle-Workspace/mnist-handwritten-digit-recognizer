
import os
from models.ann_models import CustomANNModel
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd


def main():
    model = CustomANNModel()
    data = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/train.csv"))
    X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values


if __name__ == '__main__':
    main()
