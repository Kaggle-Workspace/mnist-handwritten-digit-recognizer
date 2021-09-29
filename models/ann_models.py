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
        self.model.fit(
            X_train, y_train, epochs=epochs,
            batch_size=batch_size, verbose=1,
            # validation_data=(X_valid, y_valid)
        )
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

        self.dense_1 = Dense(input_dim=784, units=512, activation="relu",
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

        self.model = Sequential()
        self.model.add(self.dense_1)
        self.model.add(self.dense_2)
        self.model.add(self.dense_3)
        self.model.add(self.dense_4)
        # self.model.add(self.dense_5)


class ANN2(ANNDefaultModel):
    def __init__(self) -> None:
        super(ANNDefaultModel, self).__init__()

        self.dense_1 = Dense(input_dim=784, units=800, activation="relu",
                             kernel_initializer="glorot_uniform", bias_initializer="zeros")

        self.dense_2 = Dense(units=10, activation="softmax",
                             kernel_initializer="glorot_uniform", bias_initializer="zeros")

        # self.dense_3 = Dense(units=1, activation="sigmoid",
        #                     kernel_initializer="glorot_uniform", bias_initializer="zeros")

        self.model = Sequential()
        self.model.add(self.dense_1)
        self.model.add(self.dense_2)
        # self.model.add(self.dense_3)
