import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image


def set_tensorflow_max_memory_usage(gpu_memory_fraction):
    """
    By default, a tensorflow session uses all available video memory. This allocates a fraction of the complete amount
    :param gpu_memory_fraction:
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    set_session(tf.Session(config=config))


def decision(probability):
    """
    Returns true with a probability
    :param probability:
    :return: true/false
    """
    return random.random() < probability


def display_model_as_graph(model):
    """
    In an IPython notebook, displays graph visualization of a keras model
    :param model:
    :return:
    """
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    SVG(model_to_dot(model).create(prog='dot', format='svg'))


def timeDifSecondsToStr(seconds):
    """
    Creates a formated string of a time delta
    :param seconds:
    :return:
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d hours, %d minutes, %d seconds (%f total seconds)" % (h, m, s, seconds)

def prepare_one_image(image, target_size=(224,224)):
    """
    prepares an images to be fed into the network. Rescales and converts to numpy array
    :param image:
    :param target_size:
    :return:
    """
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

def predict(model, x, thresh=0.5):
    """
    Predict binary class of `x` with `model` using decision above `thresh`
    :param model: parameters of model (e.g. load from .h5 file)
    :param x: numpy array, image as numpy array in format `channel_last` (e.g. [224, 224, 3])
    :param thresh: float, if probability prediction for `x` >= thresh then classify `x` as `empty` otherwise `nempty`
    :return: int, (decision, probability prediction) where decision is 0 (=`nempty`) or 1 (=`empty`)
    """

    prob = model.predict(x, batch_size=1, steps=None, verbose=1)
    prob = prob[0][0]
    return (1, prob) if prob >= thresh else (0, prob)


def prepare_image(img, resize_side, normalize=True):
    resized = img.resize((resize_side, resize_side), Image.NEAREST)
    x = img_to_array(resized, data_format='channels_last')
    x = np.expand_dims(x, axis=0)
    if normalize:
        x /= 255.
    return x