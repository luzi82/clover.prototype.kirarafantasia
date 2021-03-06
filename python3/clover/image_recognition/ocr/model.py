from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation, Permute
from keras.layers import GaussianNoise, Concatenate, Lambda
from keras.layers import Input
from keras.layers import Reshape
from keras.layers.recurrent import GRU
from keras.models import Model
from keras import regularizers
from clover.common import PHI
from . import setting
import os
import kirarafantasia_bot.image_recognition.state as ir_state
import clover.image_recognition
from keras import backend as K
import numpy as np

def tensor_in_out(label_count,input_shape):
    tensor_in = Input(shape=input_shape, name='the_input')

    tensor = tensor_in

    tensor = Conv2D(filters=32, kernel_size=1,     padding='same', activation='elu', name='a0')(tensor)
    tensor = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='elu', name='a1')(tensor)
    tensor = Conv2D(filters=32, kernel_size=1,     padding='same', activation='elu', name='a2')(tensor)
    tensor = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='elu', name='a3')(tensor)
    tensor = Conv2D(filters=32, kernel_size=1,     padding='same', activation='elu', name='a4')(tensor)
    tensor = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='elu', name='a5')(tensor)
    tensor = Conv2D(filters=32, kernel_size=1,     padding='same', activation='elu', name='a6')(tensor)

    shape = K.int_shape(tensor)
    tensor = Reshape(target_shape=(shape[1],shape[2]*shape[3]), name='b0')(tensor)
    tensor = Dense(512,activation='elu', name='b1')(tensor)
    #tensor = Dropout(2-PHI, name='b2')(tensor)

    tensor_0 = GRU(512, return_sequences=True, go_backwards=False, name='c00')(tensor)
    tensor_1 = GRU(512, return_sequences=True, go_backwards=True , name='c01')(tensor)
    tensor = Concatenate(axis=2, name='c02')([tensor_0,tensor_1])
    tensor = Dense(512,activation='elu', name='c03')(tensor)
    #tensor = Dropout(2-PHI, name='c04')(tensor)

    tensor_0 = GRU(512, return_sequences=True, go_backwards=False, name='c10')(tensor)
    tensor_1 = GRU(512, return_sequences=True, go_backwards=True , name='c11')(tensor)
    tensor = Concatenate(axis=2, name='c12')([tensor_0,tensor_1])
    tensor = Dense(512,activation='elu', name='c13')(tensor)
    #tensor = Dropout(2-PHI, name='c14')(tensor)

    tensor_0 = GRU(512, return_sequences=True, go_backwards=False, name='c20')(tensor)
    tensor_1 = GRU(512, return_sequences=True, go_backwards=True , name='c21')(tensor)
    tensor = Concatenate(axis=2, name='c22')([tensor_0,tensor_1])
    tensor = Dense(512,activation='elu', name='c23')(tensor)
    #tensor = Dropout(2-PHI, name='c24')(tensor)

    tensor = Dense(512, activation='elu', name='d0')(tensor)
    #tensor = Dropout(2-PHI, name='d2')(tensor)
    tensor = Dense(512, activation='elu', name='d3')(tensor)
    #tensor = Dropout(2-PHI, name='d5')(tensor)
    tensor = Dense(512, activation='elu', name='d6')(tensor)
    #tensor = Dropout(2-PHI, name='d8')(tensor)
    #tensor = Dense(label_count, activity_regularizer=regularizers.l1(0.01/(label_count*input_shape[1])), name='d9')(tensor)
    #tensor = Dense(label_count, activity_regularizer=regularizers.l1(0.01), name='d9')(tensor)
    tensor = Dense(label_count, name='d9')(tensor)

    tensor = Activation('softmax', name='e0')(tensor)

    tensor_out = tensor

    return tensor_in, tensor_out

if __name__ == '__main__':
    tensor_in, tensor_out = tensor_in_out(11,(160,16,4))
    model = Model(inputs=tensor_in, outputs=tensor_out)
    model.summary()
