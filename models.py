
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Layer
import cv2
import matplotlib.pyplot as plt

def show_preprocessing_chain():
    #img = cv2.imread('./data/Center_Forward/IMG/center_2017_03_05_13_15_45_368.jpg')
    img = cv2.imread('./data/2ndTrack_RightSide/IMG/center_2017_03_18_10_17_03_591.jpg')
    #f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original RGB Image')

    plt.figure()
    img = img[50:-20,:]
    plt.imshow(img)
    plt.title('Cropped RGB Image')
    plt.figure()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(img)
    plt.figure()
    plt.imshow(h)
    plt.title('H Channel Image')

    plt.figure()
    plt.imshow(s)
    plt.title('S Channel Image')

    plt.figure()
    plt.imshow(v)
    plt.title('V Channel Image')

    plt.show()

def add_preprocessing(model):
    # Another processing step is done outside the Kera Model but should be done in Tensorflow
    # for speed reasons: from RGB to HSV to H
    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # trim image to only see section with road
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

def build_LeNet():
    model = Sequential()

    add_preprocessing(model)

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

    add_preprocessing(model)

    # Network
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    #model.add(Dropout(0.9))
    #model.add(Convolution2D(64,3,3,activation='relu'))
    #model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    return model
