import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.function_base import average
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def make_model():

    model = Sequential()

    model.add(
        Dense(
            input_shape=(784,),
            units=512,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer="zeros",
        )
    )

    model.add(
        Dense(
            units=256,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer="zeros",
        )
    )

    model.add(
        Dense(
            units=128,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer="zeros",
        )
    )

    # Multi-class classification
    model.add(
        Dense(
            units=10,
            activation="softmax",
            kernel_initializer="he_normal",
            bias_initializer="zeros",
        )
    )

    # Binary classification
    # model.add(
    #     Dense(
    #         units=1,
    #         activation="sigmoid",
    #         kernel_initializer="he_normal",
    #         bias_initializer="zeros",
    #     )
    # )
    return model


def train_model(model, X_train, X_valid, y_train, y_valid,
                epochs=10, batch_size=32,
                loss="sparse_categorical_crossentropy",
                optimizer="adam", metrics=["accuracy"]
                ):
    model.compile(loss=loss,
                  optimizer=optimizer, metrics=metrics)
    model.fit(
        X_train, y_train, epochs=epochs,
        batch_size=batch_size, verbose=1,
        validation_split=0.1,
        # validation_data=(X_valid, y_valid)
    )
    print("========Predicting on Validation data========")
    y_valid_pred = np.argmax(model.predict(X_valid), axis=-1)
    # print(y_valid_pred)
    print('Accuracy Score : ' + str(accuracy_score(y_valid, y_valid_pred)))
    print('Precision Score : ' +
          str(precision_score(y_valid, y_valid_pred, average="weighted")))
    print('Recall Score : ' +
          str(recall_score(y_valid, y_valid_pred, average="weighted")))
    print('F1 Score : ' + str(f1_score(y_valid, y_valid_pred, average="weighted")))

    print("========Evaluating on valid data========")
    results = model.evaluate(X_valid, y_valid, batch_size=batch_size)
    print("valid loss, valid acc:", results)


def predict_test_classes(model, X_test):
    y_test_pred = np.argmax(model.predict(X_test), axis=-1)
    return y_test_pred


def main():

    # Loading the data
    df_train = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/train.csv"))
    df_test = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/test.csv"))
    print(df_train.sample(10))

    X, y = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_test = df_test.values

    # Making the model
    model = make_model()
    print(model.summary())
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Training the model
    train_model(model, X_train, X_valid, y_train, y_valid)

    # Creating the submission file
    header = ["ImageId", "Label"]
    rows = []
    id = 1

    y_test_pred = predict_test_classes(model, X_test)
    print(y_test_pred)

    print("X_test shape is: ", X_test.shape)
    print("y_test_pred shape is: ", y_test_pred.shape)

    for _, pred in list(zip(X_test, y_test_pred)):
        rows.append((id, pred))
        id = id + 1

    # # Evaluate
    # print(f"Validation accuracy: {model.evaluate(X_valid, y_valid)[1]}")

    with open(
            os.path.join(os.path.dirname(__file__),
                         "../output/ann1_seq.csv"),
            "w", encoding="UTF8", newline="") as f:

        writer = csv.writer(f)

        # Write the headers
        writer.writerow(header)

        # write multiple rows
        writer.writerows(rows)


if __name__ == "__main__":
    main()
