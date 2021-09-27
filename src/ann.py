import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# from the parent directory
data = pd.read_csv("mnist-handwritten-digit-recognizer/data/train.csv")
print(data.sample(10))

X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = Sequential()

model.add(
    Dense(
        units=512,
        input_shape=(784,),
        activation="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )
)

model.add(
    Dense(
        units=256,
        activation="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )
)

model.add(
    Dense(
        units=128,
        activation="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )
)

model.add(
    Dense(
        units=10,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )
)

model.add(
    Dense(
        units=1,
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )
)


print(model.summary())
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_valid, y_valid),
    batch_size=128,
)

y_pred = model.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)

# Evaluate
print(f"Validation accuracy: {model.evaluate(X_valid, y_valid)[1]}")
