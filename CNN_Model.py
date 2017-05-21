
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Noise_functions as nf
import argparse
import sys
import math
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from skimage.util import random_noise
import numpy as np

FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    # Import data
    data = nf.blockReduce()
    inputSize = len(data[0][0])
    squareSize = math.sqrt(inputSize)
    afterMaxPool = squareSize / 2

    # Create the model
    x = tf.placeholder(tf.float32, [None, inputSize])
    W = tf.Variable(tf.zeros([inputSize, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 14 ,14, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    sess = tf.InteractiveSession()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    numOfIterations = [55, 110, 165, 220, 275, 330, 385, 440 , 495, 550]
    learningCurvePoints = [5500, 11000, 16500, 22000, 27500, 33000, 38500, 44000, 49500]
    learningCurvePercentage = []
    for i in range(0,55000,100):
        batch_xs = data[0][i: i + 99]
        batch_ys = data[2][i: i + 99]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i in learningCurvePoints:
            accuracyValue = sess.run(accuracy, feed_dict={x: data[1], y_: data[3]})
            learningCurvePercentage.append(accuracyValue)

    accuracyValue = sess.run(accuracy, feed_dict={x: data[1], y_: data[3]})
    learningCurvePercentage.append(accuracyValue)
        # Test trained model
    print (accuracyValue)
    plt.scatter(numOfIterations,learningCurvePercentage)
    plt.plot(numOfIterations, learningCurvePercentage)
    plt.show()
    print(sess.run(accuracy, feed_dict={x: data[1], y_: data[3]}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)