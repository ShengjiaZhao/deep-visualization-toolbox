__author__ = 'shengjia'

import time, os
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
sess.as_default()


def state_variable(shape, name=None):
    initial = tf.nn.relu(tf.truncated_normal(shape, stddev=0.1))
    if name is not None:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

batch_size = 100

x = tf.placeholder(tf.float32, shape=[batch_size, 784])
y_ref = tf.placeholder(tf.float32, shape=[batch_size, 10])  # Each batch contains 20 samples

with tf.name_scope('test'):
    y_weight_var = state_variable([batch_size, 10], name='y_weight')
    y_pred = tf.nn.softmax(y_weight_var, name='prediction')

train_phase = tf.placeholder(tf.bool, shape=[batch_size])

with tf.name_scope("fc1"):
    fc1_input = tf.select(train_phase, y_ref, y_pred)
    W_fc1 = weight_variable([10, 256])
    b_fc1 = bias_variable([256])
    fc1_var = state_variable([batch_size, 256], name='fc1_var')

    fc1_relu = tf.nn.relu(tf.matmul(fc1_input, W_fc1, name='fc1') + b_fc1, name='fc1_relu')
    fc1_loss = tf.reduce_sum(tf.square(tf.sub(fc1_var, fc1_relu)), name='fc1_loss') / (256*batch_size)

with tf.name_scope("fc2"):
    W_fc2 = weight_variable([256, 7 * 7 * 32])
    b_fc2 = bias_variable([7 * 7 * 32])
    fc2_var = state_variable([batch_size, 7 * 7 * 32], name='fc2_var')

    fc2_relu = tf.nn.relu(tf.matmul(fc1_var, W_fc2, name='fc2') + b_fc2, name='fc2_relu')
    fc2_loss = tf.reduce_sum(tf.square(tf.sub(fc2_var, fc2_relu)), name='fc2_loss') / (7*7*32*batch_size)

with tf.name_scope('conv1'):
    conv1_in = unpool(tf.reshape(fc2_var, [batch_size, 7, 7, 32]), name='conv1_unpool')
    W_conv1 = weight_variable([5, 5, 32, 24])
    b_conv1 = bias_variable([24])
    conv1_var = state_variable([batch_size, 14, 14, 24], name='conv1_var')

    conv1_relu = tf.nn.relu(tf.nn.conv2d(conv1_in, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    conv1_loss = tf.reduce_sum(tf.square(tf.sub(conv1_var, conv1_relu)), name='conv1_loss') / (14*14*24*batch_size)

with tf.name_scope('conv2'):
    conv2_in = unpool(conv1_var, name='conv2_unpool')
    W_conv2 = weight_variable([5, 5, 24, 1])
    b_conv2 = bias_variable([1])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    conv2_relu = tf.nn.relu(tf.nn.conv2d(conv2_in, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    conv2_loss = tf.reduce_sum(tf.square(tf.sub(x_image, conv2_relu)), name='conv2_loss') / (28*28*batch_size)

total_loss = fc1_loss + fc2_loss + conv1_loss + conv2_loss
with tf.name_scope('summary'):
    tf.scalar_summary('fc1_loss', fc1_loss)
    tf.scalar_summary('fc2_loss', fc2_loss)
    tf.scalar_summary('conv1_loss', conv1_loss)
    tf.scalar_summary('conv2_loss', conv2_loss)
    tf.scalar_summary('total_loss', total_loss)

e_learning_rate = tf.placeholder(tf.float32, shape=[])
e_step = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(total_loss,
                                                                     var_list=[fc1_var, fc2_var, conv1_var],
                                                                     name='E_optim')
m_learning_rate = tf.placeholder(tf.float32, shape=[])
m_step = tf.train.GradientDescentOptimizer(m_learning_rate).minimize(total_loss,
                                           var_list=[W_fc1, b_fc1, W_fc2, b_fc2, W_conv1, b_conv1, W_conv2, b_conv2],
                                           name='M_optim')

e_step_test = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(total_loss,
                                                                          var_list=[fc1_var, fc2_var, conv1_var, y_weight_var],
                                                                          name='E_optim_test')
summary_op = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('log/generative', sess.graph)

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if os.path.isfile('model.ckpt'):
    saver.restore(sess, "model.ckpt")
    print("Loading previous network")

e_lr = 0.01
m_lr = 0.01
test_lr = 0.01
e_step_size = 20
m_step_size = 1000
test_step_size = 100

def reinitialize():
    sess.run(fc1_var.initializer)
    sess.run(fc2_var.initializer)
    sess.run(conv1_var.initializer)
    sess.run(y_weight_var.initializer)

batch = mnist.train.next_batch(batch_size)

def test_network():
    test_batch = batch #mnist.test.next_batch(batch_size)
    reinitialize()
    test_lr = 0.01
    for e_iter in range(0, test_step_size):
        sess.run(e_step_test, feed_dict={x: test_batch[0], y_ref: test_batch[1],
                                         train_phase: [False]*batch_size, e_learning_rate: test_lr})
        test_lr *= 0.9
    truth = np.argmax(test_batch[1], 1)
    pred = np.argmax(sess.run(y_pred), 1)

    correct_count = 0
    for i in range(batch_size):
        if truth[i] == pred[i]:
            correct_count += 1
    print(str(correct_count) + " out of " + str(batch_size) + " correct")


for m_iter in range(m_step_size):
    reinitialize()
    e_lr = 0.01
    for e_iter in range(0, e_step_size):
        sess.run(e_step, feed_dict={x: batch[0], y_ref: batch[1], train_phase: [True]*batch_size, e_learning_rate: e_lr})
        e_lr *= 0.9
    for e_iter in range(0, 2):
        sess.run(e_step, feed_dict={x: batch[0], y_ref: batch[1], train_phase: [True]*batch_size, e_learning_rate: e_lr})
        sess.run(m_step, feed_dict={x: batch[0], y_ref: batch[1], train_phase: [True]*batch_size, m_learning_rate: m_lr})
    summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_ref: batch[1],
                                                  train_phase: [True]*batch_size})
    train_writer.add_summary(summary_str, m_iter)
    print("Iteration M: " + str(m_iter))
    train_writer.flush()
    #if m_iter % 10 == 0:
    m_lr *= 0.95
    if m_iter != 0 and m_iter % 20 == 0:
        test_network()
    if m_iter % 20 == 0 and e_step_size < 30:
        e_step_size += 1
    if m_iter % 50 == 0 and m_iter != 0:
        save_path = saver.save(sess, "model.ckpt")


def correctness_test():
    pass
