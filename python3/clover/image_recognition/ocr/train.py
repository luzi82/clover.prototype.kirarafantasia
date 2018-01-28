class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, minibatch_size,
                 img_w, img_h,
                 char_set,
                 absolute_max_string_len):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.char_set = char_set
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(char_set) + 1

    def get_batch(self, index, size, train):

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ocr trainer')
    parser.add_argument('epochs', type=int, help="epochs")
    args = parser.parse_args()
    
    minibatch_size = 32
    img_h = 16
    img_w = img_h*12
    
    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_w=img_w, img_h=img_h,
        char_set='0123456789',
        absolute_max_string_len=10
    )
    