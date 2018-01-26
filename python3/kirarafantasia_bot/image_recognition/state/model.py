from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.layers import GaussianNoise, Concatenate, Lambda
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model
from keras import regularizers
from clover.common import PHI
from . import setting
import os
import kirarafantasia_bot.image_recognition.state as ir_state
import clover.image_recognition
from keras import backend as K
import numpy as np

HEIGHT = setting.HEIGHT
WIDTH  = setting.WIDTH

def create_model(label_count):
    tensor_in = Input(shape=(HEIGHT,WIDTH,3))

    tensor = tensor_in
    tensor = GaussianNoise(stddev=0.10)(tensor)

    tensor0 = Flatten()(tensor)
    tensor0 = Dense(label_count, activation='elu')(tensor0)

    tensor1_shape = K.int_shape(tensor)[1:-1]+(label_count,)
    tensor1 = Dense(product(tensor1_shape), activation='elu')(tensor0)
    tensor1 = Reshape(tensor1_shape)(tensor1)
    tensor = Concatenate(axis=3)([tensor,tensor1])

    tensor = Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=32, kernel_size=(3,2), padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = MaxPooling2D(pool_size=2)(tensor)

    tensor1_shape = K.int_shape(tensor)[1:-1]+(label_count,)
    tensor1 = Dense(product(tensor1_shape), activation='elu')(tensor0)
    tensor1 = Reshape(tensor1_shape)(tensor1)
    tensor = Concatenate(axis=3)([tensor,tensor1])

    tensor = Conv2D(filters=64, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=64, kernel_size=2, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=64, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = MaxPooling2D(pool_size=2)(tensor)

    tensor1_shape = K.int_shape(tensor)[1:-1]+(label_count,)
    tensor1 = Dense(product(tensor1_shape), activation='elu')(tensor0)
    tensor1 = Reshape(tensor1_shape)(tensor1)
    tensor = Concatenate(axis=3)([tensor,tensor1])

    tensor = Conv2D(filters=128, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=128, kernel_size=2, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=128, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = MaxPooling2D(pool_size=2)(tensor)

    tensor1_shape = K.int_shape(tensor)[1:-1]+(label_count,)
    tensor1 = Dense(product(tensor1_shape), activation='elu')(tensor0)
    tensor1 = Reshape(tensor1_shape)(tensor1)
    tensor = Concatenate(axis=3)([tensor,tensor1])

    tensor = Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=256, kernel_size=3, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = MaxPooling2D(pool_size=2)(tensor)

    tensor1_shape = K.int_shape(tensor)[1:-1]+(label_count,)
    tensor1 = Dense(product(tensor1_shape), activation='elu')(tensor0)
    tensor1 = Reshape(tensor1_shape)(tensor1)
    tensor = Concatenate(axis=3)([tensor,tensor1])
    tensor = Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=256, kernel_size=(1,3), padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = GlobalAveragePooling2D()(tensor)
    tensor = BatchNormalization()(tensor)

    tensor = Dropout(2-PHI)(tensor)
    tensor = Dense(256, activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Dropout(2-PHI)(tensor)
    tensor = Dense(256, activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Dropout(2-PHI)(tensor)
    tensor = Dense(256, activation='elu', activity_regularizer=regularizers.l1(0.0001))(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Dropout(2-PHI)(tensor)
    tensor = Dense(label_count, kernel_regularizer=regularizers.l1(0.01/(label_count*256)))(tensor)

    tensor = Activation('softmax')(tensor)

    tensor_out = tensor

    model = Model(inputs=tensor_in, outputs=tensor_out)
    
    return model

def product(v_list):
    ret = 1
    for v in v_list:
        ret *= v
    return ret

if __name__ == '__main__':
    label_name_list = ir_state.get_label_list()
    label_count = len(label_name_list)
    
    model = create_model(label_count)
    model.summary()
