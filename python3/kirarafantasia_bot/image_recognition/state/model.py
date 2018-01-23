from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.layers import GaussianNoise
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from clover.common import PHI
from . import setting
import os
import kirarafantasia_bot.image_recognition.state as ir_state

HEIGHT = setting.HEIGHT
WIDTH  = setting.WIDTH

def create_model(label_count):
    tensor_in = Input(shape=(HEIGHT,WIDTH,5))
    
    tensor = tensor_in
    tensor = GaussianNoise(stddev=0.10)(tensor)
    tensor = Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=32, kernel_size=(3,2), padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = MaxPooling2D(pool_size=2)(tensor)
    tensor = Conv2D(filters=64, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=64, kernel_size=2, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=64, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = MaxPooling2D(pool_size=2)(tensor)
    tensor = Conv2D(filters=128, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=128, kernel_size=2, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=128, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = MaxPooling2D(pool_size=2)(tensor)
    tensor = Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=256, kernel_size=3, padding='valid', activation='elu')(tensor)
    tensor = Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = MaxPooling2D(pool_size=2)(tensor)
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

    model = Model(inputs=[tensor_in], outputs=tensor_out)
    
    return model

if __name__ == '__main__':
    label_name_list = ir_state.get_label_list()
    label_count = len(label_name_list)
    
    model = create_model(label_count)
    model.summary()
