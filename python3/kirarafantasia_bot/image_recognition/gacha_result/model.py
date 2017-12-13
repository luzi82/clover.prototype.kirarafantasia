from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.layers import GaussianNoise
from keras.models import Sequential
from keras import regularizers
from clover.common import PHI
import numpy as np
import clover.image_recognition

HEIGHT = 32
WIDTH  = 32

def preprocess_img(img):
    img = np.append(img,clover.image_recognition.xy_layer(WIDTH,HEIGHT),axis=2)
    return img

def create_model():
    model = Sequential()
    model.add(GaussianNoise(stddev=0.01, input_shape=(HEIGHT,WIDTH,5)))
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=1, padding='valid', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=2, padding='valid', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=1, padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=1, padding='valid', activation='elu'))
    model.add(Conv2D(filters=128, kernel_size=2, padding='valid', activation='elu'))
    model.add(Conv2D(filters=128, kernel_size=1, padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu'))
    model.add(Conv2D(filters=256, kernel_size=3, padding='valid', activation='elu'))
    model.add(Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu'))
    model.add(GlobalMaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(256, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(256, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(256, activation='elu', activity_regularizer=regularizers.l1(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(1, kernel_regularizer=regularizers.l1(0.01/256)))
    return model

