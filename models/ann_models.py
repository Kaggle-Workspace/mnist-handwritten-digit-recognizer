from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.python import keras


class CustomANNModel():

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

    def train(self, X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
        model = Sequential()
        model.add(self.dense1)
        model.add(self.dense2)
        model.add(self.dense3)
        model.add(self.dense4)
        model.add(self.dense5)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        y_pred = model.predict(X_test)
        model.evaluate(y_test,y_pred)
        