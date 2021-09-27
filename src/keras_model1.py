
import os
from models.keras_sequential_model import KerasSequentialModel
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

model = KerasSequentialModel()
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/train.csv"))


def main():
    print("Hello World")


if __name__ == '__main__':
    main()
