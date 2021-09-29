import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# from the parent directory
df_train = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "../data/train.csv"))
df_test = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "../data/test.csv"))
print(df_train.sample(10))

X, y = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values

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

# model.add(
#     Dense(
#         units=1,
#         activation="sigmoid",
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#     )
# )


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
    batch_size=128,
    validation_split=0.1,
    validation_data=(X_valid, y_valid)
)

y_pred = model.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)

# Evaluate
print(f"Validation accuracy: {model.evaluate(X_valid, y_valid)[1]}")
