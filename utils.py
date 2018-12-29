import os
import sys
from scipy import misc
from tensorflow.contrib import slim
import numpy as np
import tensorflow as tf


def get_image(src, img_size=False):
   img = misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img


def image_processing(self, filename):
    x = tf.read_file(filename)
    x_decode = tf.image.decode_jpeg(x, channels=self.channels)
    img = tf.image.resize_images(x_decode, [self.img_h, self.img_w])
    img = tf.cast(img, tf.float32) / 127.5 - 1

    return img



class ImageData:

    def __init__(self, load_size, channels, augment_flag=False, do_random_hue=False):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag
        self.do_random_hue = do_random_hue

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        if self.do_random_hue:
            img = tf.image.random_hue(img, 0.5)
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img


def load_image_np(image_path, resize=True, size_h=256, size_w=256):
    img = misc.imread(image_path, mode='RGB')
    if resize:
        img = misc.imresize(img, [size_h, size_w])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x


def load_image(filename, resize_size=None, random_hue=-1):
    x = tf.read_file(filename)
    img = tf.image.decode_jpeg(x, channels=3)
    print("load_image - img", img)
    if resize_size:
        img = tf.image.resize_images(img, [resize_size[0], resize_size[1]])
    if random_hue>=0:
        img = tf.image.random_hue(img, random_hue)
    img = tf.cast(img, tf.float32) / 127.5 - 1
    return img


def save_one_img(image, path):
    return misc.imsave(path, image)


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir




