from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

#########################################
import numpy as np
import math
from PIL import Image
# from resizeimage import resizeimage
from skimage.measure import block_reduce
from skimage.util import random_noise

FLAGS = None



def blockReduceWhiteNoise():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    MnistTrainImagesList = []
    MnistTestImagesList = []

    for pic in mnist.train.images:
        # resize the image:
        img = block_reduce(pic.reshape(28, 28), block_size=(2, 2), func=np.mean).reshape(196)
        noisyimg = random_noise(img, mode='gaussian', seed=None, clip=True)
        MnistTrainImagesList.append(noisyimg)
    MnistTrainImages = np.array(MnistTrainImagesList)

    for pic in mnist.test.images:
        # resize the image:
        img = block_reduce(pic.reshape(28, 28), block_size=(2, 2), func=np.mean).reshape(196)
        noisyimg = random_noise(img, mode='gaussian', seed=None, clip=True)
        MnistTestImagesList.append(noisyimg)
    MnistTestImages = np.array(MnistTestImagesList)

    data = (MnistTrainImages, MnistTestImages, mnist.train.labels, mnist.test.labels)

    return data


def whiteNoise():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    MnistTrainImagesList = []
    MnistTestImagesList = []

    for pic in mnist.train.images:
        noisyimg = random_noise(pic, mode='gaussian', seed=None, clip=True)
        MnistTrainImagesList.append(noisyimg)
    MnistTrainImages = np.array(MnistTrainImagesList)

    for pic in mnist.test.images:
        noisyimg = random_noise(pic, mode='gaussian', seed=None, clip=True)
        MnistTestImagesList.append(noisyimg)
    MnistTestImages = np.array(MnistTestImagesList)

    data = (MnistTrainImages, MnistTestImages, mnist.train.labels, mnist.test.labels)

    return data


def blockReduce():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    MnistTrainImagesList = []
    MnistTestImagesList = []

    for pic in mnist.train.images:
        # resize the image:
        img = block_reduce(pic.reshape(28, 28), block_size=(2, 2), func=np.mean).reshape(196)
        MnistTrainImagesList.append(img)
    MnistTrainImages = np.array(MnistTrainImagesList)

    for pic in mnist.test.images:
        img = block_reduce(pic.reshape(28, 28), block_size=(2, 2), func=np.mean).reshape(196)
        MnistTestImagesList.append(img)
    MnistTestImages = np.array(MnistTestImagesList)

    data = (MnistTrainImages, MnistTestImages, mnist.train.labels, mnist.test.labels)

    return data
