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

def state_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

batch_size = 20

x = tf.placeholder(tf.float32, shape=[batch_size, 784])
y_ref = tf.placeholder(tf.float32, shape=[batch_size, 10])  # Each batch contains 20 samples

with tf.name_scope("fc1"):
    W_fc1 = weight_variable([10, 1024])
    b_fc1 = bias_variable([1024])
    fc1_var = state_variable([batch_size, 1024])

    fc1_relu = tf.nn.relu(tf.matmul(y_ref, W_fc1, name='fc1') + b_fc1, name='fc1_relu')
    fc1_loss = tf.reduce_sum(tf.square(tf.sub(fc1_var, fc1_relu)), name='fc1_loss')

with tf.name_scope("fc2"):
    W_fc2 = weight_variable([1024, 7 * 7 * 64])
    b_fc2 = bias_variable([7 * 7 * 64])
    fc2_var = state_variable([batch_size, 7 * 7 * 64])

    fc2_relu = tf.nn.relu(tf.matmul(fc1_var, W_fc2, name='fc2') + b_fc2, name='fc2_relu')
    fc2_loss = tf.reduce_sum(tf.square(tf.sub(fc2_var, fc2_relu)), name='fc2_loss')

merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('log/generative', sess.graph)


