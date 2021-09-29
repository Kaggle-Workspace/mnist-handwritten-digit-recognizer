
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model


class ANNModel1(tf.keras.Model):

    def __init__(self):
        super(ANNModel1, self).__init__()
        self.dense_1 = Dense(input_shape=(784,), units=512, activation="relu",
                             kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense_2 = Dense(units=256, activation="relu",
                             kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense_3 = Dense(units=128, activation="relu",
                             kernel_initializer="glorot_uniform", bias_initializer="zeros")

        # np.argmax(model.predict(x), axis=-1),
        # if your model does multi-class classification (e.g. if it uses a softmax last-layer activation).
        # (model.predict(x) > 0.5).astype("int32"),
        # if your model does binary classification (e.g. if it uses a sigmoid last-layer activation).

        self.dense_4 = Dense(units=10, activation="softmax",
                             kernel_initializer="glorot_uniform", bias_initializer="zeros")

        # self.dense_5 = Dense(units=1, activation="sigmoid",
        #                     kernel_initializer="glorot_uniform", bias_initializer="zeros")

    # Dont use forward, use call
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        # x = self.dense_5(x)
        return x

    def model(self):
        x = Input(shape=(28, 28, 1))
        return Model(inputs=[x], outputs=self.call(x))

    def summary(self):
        x = Input(shape=(28, 28, 1))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class ANNModel2(tf.keras.Model):

    def __init__(self):
        super(ANNModel2, self).__init__()
        self.dense_1 = Dense(input_shape=(784,), units=800, activation="relu",
                             kernel_initializer="glorot_uniform", bias_initializer="zeros")

        self.dense_2 = Dense(units=10, activation="softmax",
                             kernel_initializer="glorot_uniform", bias_initializer="zeros")

        # self.dense_3 = Dense(units=1, activation="sigmoid",
        #                     kernel_initializer="glorot_uniform", bias_initializer="zeros")

    # Dont use forward, use call
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        # x = self.dense_3(x)
        return x

    def model(self):
        x = Input(shape=(28, 28, 1))
        return Model(inputs=[x], outputs=self.call(x))

    def summary(self):
        x = Input(shape=(28, 28, 1))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()
