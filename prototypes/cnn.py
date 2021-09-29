
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.constraints import UnitNorm
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.core import Flatten


# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main():
    data = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/train.csv"))
    X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values
    X = X.reshape(X.shape[0], 28, 28, 1)
    X = X.astype('float32')
    X /= 255
    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
    y = y.astype('int32')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    # 1st convolution layer
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
              input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # # 3rd convolution layer
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(
        units=10,
        activation='softmax'))

    print(model.summary())

    epochs = 10
    batch_size = 32
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam"
    metrics = ["accuracy"]

    model.compile(loss=loss,
                  optimizer=optimizer, metrics=metrics)
    model.fit(
        X_train, y_train, epochs=epochs,
        batch_size=batch_size, verbose=1,
        # validation_data=(X_valid, y_valid)
    )
    y_pred = np.argmax(model.predict(X_valid), axis=-1)
    print(y_pred)

    print("Evaluating on valid data")
    results = model.evaluate(X_valid, y_valid, batch_size=batch_size)
    print("valid loss, valid acc:", results)


if __name__ == '__main__':
    main()
