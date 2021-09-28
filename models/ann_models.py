from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.python import keras


class ANNDefaultModel():

    def __init__(self) -> None:
        pass

    def train(self, X_train, X_valid, y_train, y_valid,
              epochs=10, batch_size=32,
              loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"]
              ):
        self.model.compile(loss=loss,
                           optimizer=optimizer, metrics=metrics)
        self.model.fit(X_train, y_train, epochs=epochs,
                       batch_size=batch_size, verbose=1)
        y_pred = self.model.predict_classes(X_valid)
        print(y_pred)

        print("Evaluating on valid data")
        results = self.model.evaluate(X_valid, y_valid, batch_size=batch_size)
        print("valid loss, valid acc:", results)

    def predict_test_classes(self, X_test):
        y_pred = self.model.predict_classes(X_test)
        return y_pred


class ANN1(ANNDefaultModel):

    def __init__(self) -> None:
        super(ANNDefaultModel, self).__init__()

        self.dense1 = Dense(input_dim=784, units=512, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense2 = Dense(units=256, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense3 = Dense(units=128, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense4 = Dense(units=10, activation="softmax",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")

        # This is classification NOT regression
        # self.dense5 = Dense(units=1, activation="sigmoid",
        #                     kernel_initializer="glorot_uniform", bias_initializer="zeros")

        self.model = Sequential()
        self.model.add(self.dense1)
        self.model.add(self.dense2)
        self.model.add(self.dense3)
        self.model.add(self.dense4)
        # self.model.add(self.dense5)


class ANN2(ANNDefaultModel):
    def __init__(self) -> None:
        super(ANNDefaultModel, self).__init__()

        self.dense1 = Dense(input_dim=784, units=800, activation="relu",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.dense2 = Dense(units=10, activation="softmax",
                            kernel_initializer="glorot_uniform", bias_initializer="zeros")

        # This is classification NOT regression
        # self.dense3 = Dense(units=1, activation="sigmoid",
        #                     kernel_initializer="glorot_uniform", bias_initializer="zeros")

        self.model = Sequential()
        self.model.add(self.dense1)
        self.model.add(self.dense2)
        # self.model.add(self.dense3)
