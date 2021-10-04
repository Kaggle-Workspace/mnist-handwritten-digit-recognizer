import warnings
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from models.cnn_models import CNN1
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical, plot_model


# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def plot_image_grid(X, y, nrows, ncols, figname="sample.png"):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for i in range(nrows * ncols):
        # axes[row, col] = plt.imshow(X_train[count])
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(X[i])
        plt.xlabel(np.argmax(y[i]))
    plt.show()
    plt.savefig(figname)


def preprocess_image_input(X):
    X = X.reshape(X.shape[0], 28, 28, 1)
    X = X.astype('float32')

    # RGB values are usually stored as integers to save memory. But doing math on colors is usually done in float because it's easier, more powerful, and more precise. The act of converting floats to integers is called "Quantization", and it throws away precision.

    # Typically, RGB values are encoded as 8-bit integers, which range from 0 to 255. It's an industry standard to think of 0.0f as black and 1.0f as white (max brightness). To convert [0, 255] to [0.0f, 1.0f] all you have to do is divide by 255.0f.
    X /= 255

    return X


def predict_test_classes(model, X_test):
    y_test_pred = np.argmax(model.predict(X_test), axis=-1)
    return y_test_pred


def main():
    df_train = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../input/train.csv"))
    df_test = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../input/test.csv"))

    X, y = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    X_test = df_test.iloc[0:, 0:].values

    X = preprocess_image_input(X)
    X_test = preprocess_image_input(X_test)

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
    plot_image_grid(X_train, y_train, nrows,
                    ncols, figname=os.path.join(os.path.dirname(__file__),
                                                "../resources/images/training_image_lables.png"))

    model = CNN1()
    model.build(input_shape=X.shape)
    # model.train(X_train, X_valid, y_train,
    #             y_valid, epochs=10, batch_size=32)

    # y_pred = model.predict_test_classes(X_test)
    # print(y_pred)
    epochs = 10
    batch_size = 32
    loss = "categorical_crossentropy",
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
    plot_image_grid(X_test, to_categorical(y_test_pred), nrows,
                    ncols, figname=os.path.join(os.path.dirname(__file__),
                                                "../resources/images/test_image_predictions.png")

                    )
    with open(
            os.path.join(os.path.dirname(__file__),
                         "../output/cnn1_func.csv"),
            "w", encoding="UTF8", newline="") as f:

        writer = csv.writer(f)

        # Write the headers
        writer.writerow(header)

        # write multiple rows
        writer.writerows(rows)

    dot_img_file = os.path.join(os.path.dirname(__file__),
                                "../resources/images/cnn1_func.png")
    plot_model(model, to_file=dot_img_file, show_shapes=True)


if __name__ == '__main__':
    main()
