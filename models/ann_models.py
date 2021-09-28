from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.python import keras


class ANN1():

    def __init__(self) -> None:
        self.dense1 = Dense(input_dim=784, units=512, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense2 = Dense(units=256, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense3 = Dense(units=128, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense4 = Dense(units=10, activation="softmax",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense5 = Dense(units=1, activation="sigmoid",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")

    def train(self, X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
        model = Sequential()
        model.add(self.dense1)
        model.add(self.dense2)
        model.add(self.dense3)
        model.add(self.dense4)
        model.add(self.dense5)
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=epochs,
                  batch_size=batch_size, verbose=1)
        y_pred = model.predict(X_test)
        print(y_pred)

        print("Evaluating on test data")
        results = model.evaluate(X_test, y_test, batch_size=batch_size)
        print("test loss, test acc:", results)


class ANN2():
    def __init__(self) -> None:
        self.dense1 = Dense(input_dim=784, units=800, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense2 = Dense(units=10, activation="softmax",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense3 = Dense(units=1, activation="sigmoid",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")

    def train(self, X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
        model = Sequential()
        model.add(self.dense1)
        model.add(self.dense2)
        model.add(self.dense3)
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=epochs,
                  batch_size=batch_size, verbose=1)
        y_pred = model.predict(X_test)
        print(y_pred)

        print("Evaluating on test data")
        results = model.evaluate(X_test, y_test, batch_size=batch_size)
        print("test loss, test acc:", results)
