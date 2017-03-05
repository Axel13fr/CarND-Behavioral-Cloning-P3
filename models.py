
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

def build_LeNet():
    model = Sequential()
    ## Preprocessing:
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    # Network
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.9))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.75))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.75))
    model.add(Flatten())
    #model.add(Dense(120))
    #model.add(Dropout(0.5))
    model.add(Dense(80))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build_Nvidia():
    model = Sequential()
    ## Preprocessing:
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    # Network
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    #model.add(Dropout(0.9))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
