from warnings import filters
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Flatten


class CNNDefaultModel():

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


class CNN1(CNNDefaultModel):

    def __init__(self) -> None:
        super(CNNDefaultModel, self).__init__()

        self.conv2d_1 = Conv2D(input_shape=(784,), filters=64, kernel_size=(
            1, 1), activation="relu", padding="valid")
        self.conv2d_2 = Conv2D(filters=32, kernel_size=(
            1, 1), activation="relu", padding="valid")
        self.pool2d_1 = MaxPool2D(pool_size=(2, 2))
        self.flatten_1 = Flatten()
        self.dropout_1 = Dropout(0.2)
        self.fc_1 = Dense(units=10, activation="softmax",
                          kernel_initializer="he_uniform", bias_initializer="zeros")

        self.model = Sequential()
        self.model.add(self.conv2d_1)
        self.model.add(self.conv2d_2)
        self.model.add(self.pool2d_1)
        self.model.add(self.flatten_1)
        self.model.add(self.dropout_1)
        self.model.add(self.fc_1)
        print(self.model.summary())
