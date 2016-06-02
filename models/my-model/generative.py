__author__ = 'shengjia'

import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1)
# h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
# h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_conv2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('log/train', sess.graph)
test_writer = tf.train.SummaryWriter('log/test')

sess.run(tf.initialize_all_variables())


train_iter = []
train_result = []
test_iter = []
test_result = []

start_time = time.time()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run([accuracy], feed_dict={x:batch[0], y_: batch[1]})[0]
        print("step %d, training accuracy %g"%(i, train_accuracy))
        elapsed_time = time.time() - start_time
        start_time = time.time()
        if i != 0:
            print("Step time: " + str(elapsed_time))
        train_iter.append(i)
        train_result.append(train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    if i % 200 == 0:
        start_time = time.time()
        test_accuracy = sess.run([accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})[0]
        print("test accuracy %g" % test_accuracy)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print("Testing time: " + str(elapsed_time))
        test_iter.append(i)
        test_result.append(test_accuracy)
