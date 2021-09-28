
import csv
import os

from scipy.sparse.construct import rand
from tensorflow.python.keras.backend import shape
from models.ann_models import ANN1, ANN2
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_control_flow_ops import ref_next_iteration


def main():
    df_train = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/train.csv"))
    df_test = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/test.csv"))

    X, y = df_train.iloc[1:, 1:].values, df_train.iloc[1:, 0].values
    print("The shape of the input is: ", X.shape, y.shape)
    X_train, X_valid, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_test = df_test.iloc[1:, 0:].values

    # model = ANN1()
    model = ANN2()
    model.train(X_train, X_valid, y_train,
                y_test, epochs=10, batch_size=32)

    y_pred = model.predict(X_test)
    print(y_pred)

    header = ["ImageId", "Label"]
    rows = []
    id = 1
    for _, pred in list(zip(X_valid, y_pred)):
        rows.append((id, pred))
        id = id + 1
    # print(rows)

    with open(
            os.path.join(os.path.dirname(__file__), "../data/my_submission.csv"),
            "w", encoding="UTF8", newline="") as f:

        writer = csv.writer(f)

        # Write the headers
        writer.writerow(header)

        # write multiple rows
        writer.writerows(rows)


if __name__ == '__main__':
    main()
