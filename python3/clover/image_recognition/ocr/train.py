import captcha.image as captcha_image
import os
from . import draw_text
import time
from . import model
from keras.layers import Input, Lambda
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model
import clover.common
import numpy as np
import random
import cv2

class TextImageGenerator():

    def __init__(self, minibatch_size,
                 img_h0, char_w0,
                 img_h1_min, img_h1_max,
                 img_w2, img_h2, img_w2b,
                 char_set,
                 max_word_len):

        assert(img_w2>char_w0*img_h2/img_h0*max_word_len)

        self.minibatch_size = minibatch_size
        self.img_w0 = round(img_w2 * img_h0 / img_h2)
        self.img_h0 = img_h0
        self.char_w0 = char_w0
        self.img_h1_min = img_h1_min
        self.img_h1_max = img_h1_max
        self.img_w2 = img_w2
        self.img_h2 = img_h2
        self.img_w2b = img_w2b
        self.char_set = char_set
        self.blank_label = len(char_set)
        self.max_word_len = max_word_len
        self.draw_text = draw_text.DrawText()

    def get_batch(self, size):
        X_data = np.zeros([size, self.img_w2+self.img_w2b*2, self.img_h2, 4])
        labels = np.ones([size, self.max_word_len*2])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])

        X_data[:,:,:,3]=-1

        for i in range(size):
            word_len = random.randint(0,self.max_word_len)
            word = ''.join(random.choice(self.char_set) for _ in range(word_len))

            h0 = self.img_h0
            w0 = self.char_w0 * word_len
            w0 = min(w0, self.img_w0)
            w0 = random.randint(w0, self.img_w0)
            img = self.draw_text.create_image(word, w0, h0)
            img = np.asarray(img,dtype=np.float32)
            img = ((img/255)*2)-1
            assert(img.shape==(h0,w0,3))
            assert(np.amax(img)<=1)
            assert(np.amin(img)>=-1)
            
            h1 = random.randint(self.img_h1_min, self.img_h1_max)
            w1 = round(w0*h1/h0)
            img = cv2.resize(img,dsize=(w1,h1))
            assert(img.shape==(h1,w1,3))
            
            h2 = self.img_h2
            w2 = round(w0*h2/h0)
            w2 = min(w2,self.img_w2)
            img = cv2.resize(img,dsize=(w2,h2))
            assert(img.shape==(h2,w2,3))

            img = np.transpose(img,(1,0,2))
            assert(img.shape==(w2,h2,3))

            X_data[i, self.img_w2b:self.img_w2b+w2, :, 0:3] = img
            X_data[i, self.img_w2b:self.img_w2b+w2, :, 3] = 1
            labels[i, 0:word_len] = text_to_labels(word, self.char_set)
            input_length[i] = w2
            label_length[i] = word_len
            #source_str.append(word)
        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_batch(self):
        while 1:
            ret = self.get_batch(self.minibatch_size)
            yield ret

    def get_label_count(self):
        return len(char_set) + 1

    def get_model_input_shape(self):
        return (self.img_w2+self.img_w2b*2, self.img_h2, 4)

def ctc_lambda_func_factory(img_w2b):
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, img_w2b:-img_w2b, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    return ctc_lambda_func

def text_to_labels(text, chars):
    ret = []
    for char in text:
        ret.append(chars.find(char))
    return ret

OUTPUT_DIR = os.path.join('image_recognition','model','ocr')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ocr trainer')
    parser.add_argument('epochs', type=int, help="epochs")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    clover.common.reset_dir(output_dir)

    timestamp = str(int(time.time()))
    minibatch_size = 32
    train_count = 3200
    val_count = 320
    char_set='0123456789'
    max_word_len = 10
    
    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_h0=40, char_w0=20,
        img_h1_min=6, img_h1_max=40,
        img_w2=100, img_h2=16, img_w2b=8,
        char_set=char_set,
        max_word_len=max_word_len
    )
    
    tensor_in, tensor_out = model.tensor_in_out(img_gen.get_label_count(),img_gen.get_model_input_shape())

    labels = Input(name='the_labels', shape=[max_word_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func_factory(8), output_shape=(1,), name='ctc')([tensor_out, labels, input_length, label_length])

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[tensor_in, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(output_dir,'weight.{epoch:06d}.hdf5'))
    model_best = ModelCheckpoint(filepath=os.path.join(output_dir,'weight.best.hdf5'))
    csv_logger = CSVLogger(filename=os.path.join(output_dir,'log.csv'))

    model.fit_generator(generator=img_gen.next_batch(),
                        steps_per_epoch=train_count // minibatch_size,
                        epochs=args.epochs,
                        validation_data=img_gen.next_batch(),
                        validation_steps=val_count // minibatch_size,
                        callbacks=[model_checkpoint, model_best, csv_logger],
                        verbose=2
                        )
