from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, ReLU


class CNNDefaultModel():

    def __init__(self) -> None:
        pass


class CNN1(CNNDefaultModel):

    def __init__(self) -> None:
        super(CNNDefaultModel, self).__init__()
        
        self.conv2d1 = Conv2D() 

        