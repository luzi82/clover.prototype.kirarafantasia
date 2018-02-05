import captcha.image as captcha_image
import os
from . import draw_text

class TextImageGenerator(keras.callbacks.Callback):

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
        self.char_set = char_set
        self.blank_label = len(char_set)
        self.max_word_len = max_word_len
        self.draw_text = draw_text.DrawText()

    def get_output_size(self):
        return len(char_set) + 1

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
            w0 = self.char_w * word_len
            w0 = min(width, self.img_w0)
            w0 = random.randint(w0, self.img_w0)
            img = self.draw_text.create_image(word, w0, h0)
            img = np.asarray(img,dtype=np.float32)
            img = ((img/255)*2)-1
            assert(img.shape==(h0,w0,3))
            assert(np.amax(img)<=1)
            assert(np.amin(img)>=-1)
            
            h1 = random.randint(self.img_h1_min, self.img_h1_max)
            w1 = round(w0*h1/h0)
            img = cv2.resize(img,dsize(w1,h1))
            assert(img.shape==(h1,w1,3))
            
            h2 = self.img_h2
            w2 = round(w0*h2/h0)
            w2 = min(w2,self.img_w)
            img = cv2.resize(img,dsize(w2,h2))
            assert(img.shape==(h2,w2,3))

            img = np.transpose(img,(1,0,2))
            assert(img.shape==(w2,h2,3))

            X_data[i, self.img_w2b:self.img_w2b+w2, :, 0:3] = img
            X_data[i, self.img_w2b:self.img_w2b+w2, :, 3] = 1
            labels[i, 0:word_len] = text_to_labels(word)
            input_length[i] = w2
            label_length[i] = word_len
            source_str.append(word)
        inputs = {
            'input_img': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.minibatch_size)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.minibatch_size)
            yield ret

OUTPUT_DIR = os.path.join('image_recognition','model','ocr')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ocr trainer')
    parser.add_argument('epochs', type=int, help="epochs")
    args = parser.parse_args()
    
    timestamp = str(int(time.time()))
    minibatch_size = 32
    img_h = 16
    img_w = img_h*12
    train_count = 3200
    val_count = 320
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    
    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_h0=40, char_w0=20,
        img_h1_min=6, img_h1_max=40,
        img_w2=100, img_h2=16, img_w2b=8,
        char_set='0123456789',
        max_word_len=10
    )
    