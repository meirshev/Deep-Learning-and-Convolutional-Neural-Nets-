
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import sys
import Noise_functions as nf

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

#########################################
import numpy as np
from PIL import Image
# from resizeimage import resizeimage
from skimage.measure import block_reduce
from skimage.util import random_noise
# from scipy.ndimage import gaussian_filter


FLAGS = None


# plt.imshow(mnist.train.images[1].reshape([28, 28]), cmap='gray')
# plt.show()
# plt.savefig("fig.png")
def main(_):

    #Generate data:
    data = nf.blockReduce()


    # Create the model
    x = tf.placeholder(tf.float32, [None, 196])
    W = tf.Variable(tf.zeros([196, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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
    # plt.scatter(numOfIterations,learningCurvePercentage)
    # plt.plot(numOfIterations, learningCurvePercentage)
    # plt.show()
    # print(sess.run(accuracy, feed_dict={x: data[1], y_: data[3  ]}))
    xs = np.array([20, 50, 100, 200, 250, 500])
    ysL = np.array([0.9112, 0.9246, 0.9237, 0.919, 0.9173, 0.8992])
    plt.scatter(xs,ysL)
    plt.plot(xs, ysL)
    plt.show()
    print(sess.run(accuracy, feed_dict={x: data[1], y_: data[3  ]}))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

