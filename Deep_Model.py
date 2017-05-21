
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Noise_functions as nf
import argparse
import sys
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Generate data
    data = nf.blockReduce()

    sizeOfImg = len(data[0][0])

    # Create the model
    x = tf.placeholder(tf.float32, [None, sizeOfImg])
    W_1 = tf.Variable(tf.random_normal([sizeOfImg, sizeOfImg]))
    b_1 = tf.Variable(tf.random_normal([sizeOfImg]))
    hidden_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)


    W_2 = tf.Variable(tf.random_normal([sizeOfImg, sizeOfImg]))
    b_2 = tf.Variable(tf.random_normal([sizeOfImg]))
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, W_2) + b_2)

    W_3 = tf.Variable(tf.random_normal([sizeOfImg, 10]))
    b_3 = tf.Variable(tf.random_normal([10]))
    y = tf.matmul(hidden_2, W_3) + b_3
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

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
    print (accuracyValue )
    # plt.scatter(numOfIterations,learningCurvePercentage)
    # plt.plot(numOfIterations, learningCurvePercentage)
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)