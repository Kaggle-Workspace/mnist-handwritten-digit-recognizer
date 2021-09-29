
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPool2D)


class CNN1(tf.keras.Model):

    def __init__(self) -> None:
        super(CNN1, self).__init__()

        # 1st convolution layer
        self.conv2d_1 = Conv2D(
            filters=32, kernel_size=(3, 3), activation='relu')
        self.conv2d_2 = Conv2D(
            filters=32, kernel_size=(3, 3), activation='relu')
        self.maxpool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.batch_norm_1 = BatchNormalization()
        self.dropout_1 = Dropout(0.5)

        # Second convolution layer
        self.conv2d_3 = Conv2D(64, (3, 3), activation='relu')
        self.conv2d_4 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.batch_norm_2 = BatchNormalization()
        self.dropout_2 = Dropout(0.5)

        self.flatten = Flatten()
        self.dense_1 = Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv2d_1(inputs)
        x = self.conv2d_2(x)
        x = self.maxpool_1(x)
        x = self.batch_norm_1(x, training=training)
        x = self.dropout_1(x, training=training)

        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.maxpool_2(x)
        x = self.batch_norm_2(x, training=training)
        x = self.dropout_2(x, training=training)

        x = self.flatten(x)
        x = self.dense_1(x)
        return x
