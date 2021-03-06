from captcha.image import ImageCaptcha
import os
import argparse
import cv2
import threading

def get_font_list():
    font_list = []
    font_search_path = os.path.join('resource_set','font_set')
    for dirpath, _, filenames in os.walk(font_search_path):
        for filename in filenames:
            good = False
            good = good or filename.endswith('.ttf')
            if not good:
                continue
            file_path = os.path.join(dirpath,filename)
            font_list.append(file_path)
    return font_list

class DrawText:

    def __init__(self):
        self.image_captcha = ImageCaptcha(fonts=get_font_list())
        self.image_captcha._enable_back_text = False
        self.image_captcha._enable_background_noise = False
        self.image_captcha._enable_noise_bg = False
        self.image_captcha._enable_noise_dot = False
        self.image_captcha._enable_noise_curve = False
        self.image_captcha._enable_panda = True
        self.lock = threading.Lock()

    def create_image(self, text, width, height):
        with self.lock:
            self.image_captcha.set_size(width, height)
            return self.image_captcha.generate_image(text)

if __name__ == '__main__':
    import pylab
    
    parser = argparse.ArgumentParser(description='draw text')
    parser.add_argument('text', help="text")
    parser.add_argument('width', type=int, help="text")
    parser.add_argument('height', type=int, help="text")
    args = parser.parse_args()
    
    draw_text = DrawText()
    img = draw_text.create_image(args.text, args.width, args.height)
    pylab.imshow(img)
    pylab.show()
