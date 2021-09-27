from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.python import keras


class KerasSequentialModel():

    def __init__(self) -> None:
        self.dense1 = Dense(input_dim=(784,), units=512, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense2 = Dense(units=256, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense3 = Dense(units=128, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense4 = Dense(units=10, activation="softmax",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense5 = Dense(units=1, activation="sigmoid",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return x

