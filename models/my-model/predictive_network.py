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
    initial = tf.nn.relu(tf.truncated_normal(shape, stddev=1.0))
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
x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('test'):
    y_weight_var = state_variable([batch_size, 10], name='y_weight')
    y_pred = tf.nn.softmax(y_weight_var, name='prediction')

train_phase = tf.placeholder(tf.bool, shape=[batch_size])
transform_dim = 2
transform_var = state_variable([batch_size, transform_dim], name='transform_var')
with tf.name_scope("fc1"):
    fc1_input = tf.concat(1, [tf.select(train_phase, y_ref, y_pred), transform_var])
    W_fc1 = weight_variable([10+transform_dim, 28*28*8])
    b_fc1 = bias_variable([28*28*8])
    fc1_var = state_variable([batch_size, 28*28*8], name='fc1_var')

    fc1_relu = tf.nn.relu(tf.matmul(fc1_input, W_fc1, name='fc1') + b_fc1, name='fc1_relu')
    fc1_loss = tf.reduce_sum(tf.square(tf.sub(fc1_var, fc1_relu)), name='fc1_loss') / (28*28*8*batch_size)
    fc1_reg = -tf.log(tf.reduce_sum(fc1_relu) / (28*28*8*batch_size) + 1)
    fc1_weight_loss = tf.reduce_sum(tf.square(W_fc1)) / (10*28*28*8+transform_dim*28*28*8)

with tf.name_scope('conv2'):
    conv2_in = tf.reshape(fc1_var, (batch_size, 28, 28, 8)) #unpool(conv1_var, name='conv2_unpool')
    W_conv2 = weight_variable([5, 5, 8, 1])
    b_conv2 = bias_variable([1])

    conv2_relu = tf.nn.sigmoid(tf.nn.conv2d(conv2_in, W_conv2, strides=[1, 1, 1, 1], padding='SAME'))
    conv2_loss = tf.reduce_sum(tf.square(tf.sub(x_image, conv2_relu)), name='conv2_loss') / (28*28*batch_size)

# Some hints:
# 1. be very careful about the setting of loss coeff. Upper layers should set high loss
# 2. When the internal nodes become Gaussians with very small variance, does this mean that the network is almost exact
# but still convex?
# TODO: Plot posterior activation for internal nodes


total_train_loss = fc1_loss + conv2_loss # + #conv1_loss + conv2_loss
total_test_loss = fc1_loss * 100 + conv2_loss

with tf.name_scope('summary'):
    tf.scalar_summary('fc1_loss', fc1_loss)
    # tf.scalar_summary('fc2_loss', conv1_loss)
    tf.scalar_summary('fc3_loss', conv2_loss)
    tf.scalar_summary('total_loss', total_train_loss)
    # tf.scalar_summary('fc4_loss', fc4_loss)
    '''
    tf.scalar_summary('fc1_weight_loss', fc1_weight_loss)
    tf.scalar_summary('fc2_weight_loss', fc2_weight_loss)
    tf.scalar_summary('fc3_weight_loss', fc3_weight_loss)
    tf.scalar_summary('fc1_reg', fc1_reg)
    tf.scalar_summary('fc2_reg', fc2_reg)
    #tf.scalar_summary('conv1_loss', conv1_loss)
    #tf.scalar_summary('conv2_loss', conv2_loss)

    tf.histogram_summary('fc1_weight_hist', W_fc1)
    tf.histogram_summary('fc2_weight_hist', W_fc2)
    tf.histogram_summary('fc3_weight_hist', W_fc3)
    tf.histogram_summary('fc4_weight_hist', W_fc4)'''

e_learning_rate = tf.placeholder(tf.float32, shape=[])
e_step = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(total_train_loss,
                                                                     var_list=[fc1_var, transform_var],
                                                                     name='E_optim')
m_learning_rate = tf.placeholder(tf.float32, shape=[])
m_step = tf.train.GradientDescentOptimizer(m_learning_rate).minimize(total_train_loss,
                                           var_list=[W_fc1, b_fc1, W_conv2, b_conv2],
                                           name='M_optim')

e_step_test = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(total_test_loss,
                                                                          var_list=[fc1_var, transform_var, y_weight_var],
                                                                          name='E_optim_test')
summary_op = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('log/generative', sess.graph)

y_vis = tf.placeholder(tf.float32, shape=[1, 12])
with tf.name_scope("visualization"):
    vis_fc1_relu = tf.nn.relu(tf.matmul(y_vis, W_fc1) + b_fc1)
    # vis_conv1_relu = tf.nn.relu(tf.nn.conv2d(unpool(tf.reshape(vis_fc1_relu, (1, 7, 7, 32))), W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    image_out = tf.nn.sigmoid(tf.nn.conv2d(tf.reshape(vis_fc1_relu, (1, 28, 28, 8)), W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if os.path.isfile('model.ckpt'):
    saver.restore(sess, "model.ckpt")
    print("Loading previous network")

e_lr_init = 3000
m_lr = 5
test_lr_init = 800
e_step_size = 30
m_step_size = 10000
test_step_size = 100


def reinitialize():
    sess.run(fc1_var.initializer)
    # sess.run(conv1_var.initializer)
    # sess.run(conv2_var.initializer)
    # sess.run(conv1_var.initializer)
    sess.run(y_weight_var.initializer)


def test_network():
    test_batch = mnist.test.next_batch(batch_size)
    reinitialize()
    test_lr = test_lr_init
    for e_iter in range(0, test_step_size):
        sess.run(e_step_test, feed_dict={x: test_batch[0], y_ref: test_batch[1],
                                         train_phase: [False]*batch_size, e_learning_rate: test_lr})
        test_lr *= 0.9
        # print(sess.run(fc1_var)[0, 0:10])
    print("Loss is " + str(sess.run(fc1_loss, feed_dict={x: test_batch[0], y_ref: test_batch[1],
                                            train_phase: [False]*batch_size})))
    # print(sess.run(y_weight_var))
    sess.run(y_weight_var.initializer)
    print("Loss is " + str(sess.run(fc1_loss, feed_dict={x: test_batch[0], y_ref: test_batch[1],
                                           train_phase: [False]*batch_size})))
    truth = np.argmax(test_batch[1], 1)
    pred = np.argmax(sess.run(y_weight_var), 1)
    print("Loss is " + str(sess.run(total_test_loss, feed_dict={x: test_batch[0], y_ref: test_batch[1],
                                            train_phase: [False]*batch_size})))
    print(np.average(np.abs(sess.run(fc1_relu, feed_dict={x: test_batch[0], y_ref: test_batch[1],
                                            train_phase: [False]*batch_size}))))
    print(np.average(np.abs(sess.run(W_fc2))))
    print(np.average(np.abs(sess.run(W_fc3))))
    correct_count = 0
    for i in range(batch_size):
        if truth[i] == pred[i]:
            correct_count += 1
    print(str(correct_count) + " out of " + str(batch_size) + " correct")

plt.ion()
plt.show()
vis_index = 0

# TODO: trying using only final predictive loss

import random
def visualize():
    global vis_index
    for i in range(10):
        plt.subplot(3, 4, i)
        input_label = [0] * 12
        input_label[i] = 1
        for j in range(transform_dim):
            input_label[10 + j] = random.random() * (transform_var_ub[j] - transform_var_lb[j]) + transform_var_lb[j]
        vis_result = sess.run(image_out, feed_dict={y_vis: [input_label]})
        plt.imshow(vis_result[0, :, :, 0], cmap=plt.get_cmap('Greys'))
    plt.draw()
    plt.savefig('vis/image' + str(vis_index) + '.png')
    vis_index += 1
# visualize()

def visualize_all():
    plt.ioff()
    for i in range(10):
        canvas = np.zeros((28*10, 28*10), np.float)
        for i_step in range(10):
            for j_step in range(10):
                input_label = [0] * 12
                input_label[i] = 1
                input_label[10] = float(i_step) / 10.0 * (transform_var_ub[0] - transform_var_lb[0]) + transform_var_lb[0]
                input_label[11] = float(j_step) / 10.0 * (transform_var_ub[1] - transform_var_lb[1]) + transform_var_lb[1]
                vis_result = sess.run(image_out, feed_dict={y_vis: [input_label]})
                canvas[i_step*28:i_step*28+28, j_step*28:j_step*28+28] = vis_result[:, :, 0]
        plt.imshow(canvas, cmap=plt.get_cmap('Greys'))
        plt.show()

def plot_conv1_activation():
    activation = np.reshape(sess.run(fc1_var), (batch_size, 28, 28, 8))
    plt.ioff()
    for i in range(8):
        plt.subplot(3, 3, i)
        plt.imshow(np.clip(activation[0, :, :, i], 0, 1), interpolation='none', cmap=plt.get_cmap('Greys'))
    plt.show()
    plt.ion()

transform_var_ub = np.zeros(2)
transform_var_lb = np.zeros(2)
for m_iter in range(m_step_size):
    batch = mnist.train.next_batch(batch_size)
    #if m_iter % 5 == 0:
        #test_network()
    use_label = True
    reinitialize()
    e_lr = e_lr_init
    res = []
    for e_iter in range(0, e_step_size):
        loss = sess.run([e_step, conv2_loss], feed_dict={x: batch[0], y_ref: batch[1], train_phase: [use_label]*batch_size, e_learning_rate: e_lr})[1]
        if e_iter > 15:
            e_lr *= 0.8
        res.append(loss)
    # plt.ioff()
    # plt.plot(res)
    # plt.show()
    # plot_conv1_activation()
        # print(sess.run(fc1_var)[0, 0:10])
    transform_value = sess.run(transform_var)
    for b in range(batch_size):
        for i in range(transform_dim):
            transform_var_ub[i] = max(transform_value[b, i], transform_var_ub[i])
            transform_var_lb[i] = min(transform_value[b, i], transform_var_lb[i])
    transform_var_new_ub = transform_var_lb + 0.99 * (transform_var_ub - transform_var_lb)  # Decay this by 0.01
    transform_var_new_lb = transform_var_ub - 0.99 * (transform_var_ub - transform_var_lb)  # Decay this by 0.01
    transform_var_ub = transform_var_new_ub
    transform_var_lb = transform_var_new_lb

    for e_iter in range(0, 2):
        sess.run(e_step, feed_dict={x: batch[0], y_ref: batch[1], train_phase: [use_label]*batch_size, e_learning_rate: e_lr})
        sess.run(m_step, feed_dict={x: batch[0], y_ref: batch[1], train_phase: [use_label]*batch_size, m_learning_rate: m_lr})
    summary_str, loss_result = sess.run([summary_op, total_train_loss], feed_dict={x: batch[0], y_ref: batch[1],
                                                  train_phase: [use_label]*batch_size})

    train_writer.add_summary(summary_str, m_iter)
    print("Iteration M: " + str(m_iter) + " with loss " + str(loss_result))
    train_writer.flush()
    if m_iter % 60 == 0:
        m_lr *= 0.95

    if m_iter % 20 == 0 and e_step_size < 30:
        e_step_size += 1
    if m_iter % 50 == 0 and m_iter != 0:
        save_path = saver.save(sess, "model.ckpt")
    if m_iter % 20 == 0:
        visualize()





