__author__ = 'shengjia'

import time, os
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cPickle
import numpy as np
import tensorflow as tf
from cifar import *


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

dataset = DatasetCIFAR()
x = tf.placeholder(tf.float32, shape=[batch_size, 32*32*3])
y_ref = tf.placeholder(tf.float32, shape=[batch_size, 10])  # Each batch contains 20 samples
x_image = tf.reshape(x, [-1, 32, 32, 3])

with tf.name_scope('test'):
    y_weight_var = state_variable([batch_size, 10], name='y_weight')
    y_pred = tf.nn.softmax(y_weight_var, name='prediction')

train_phase = tf.placeholder(tf.bool, shape=[batch_size])
transform_dim = 2
transform_var = state_variable([batch_size, transform_dim], name='transform_var')
with tf.name_scope("fc1"):
    fc1_input = tf.concat(1, [tf.select(train_phase, y_ref, y_pred), transform_var])
    W_fc1 = weight_variable([10+transform_dim, 256])
    b_fc1 = bias_variable([256])
    fc1_var = state_variable([batch_size, 256], name='fc1_var')

    fc1_relu = tf.nn.relu(tf.matmul(fc1_input, W_fc1, name='fc1') + b_fc1, name='fc1_relu')
    fc1_loss = tf.reduce_sum(tf.square(tf.sub(fc1_var, fc1_relu)), name='fc1_loss') / (256*batch_size)
    fc1_reg = -tf.log(tf.reduce_sum(fc1_relu) / (256*batch_size) + 1)
    fc1_weight_loss = tf.reduce_sum(tf.square(W_fc1)) / (10*256+transform_dim*256)

with tf.name_scope("fc2"):
    W_fc2 = weight_variable([256, 256])
    b_fc2 = bias_variable([256])
    fc2_var = state_variable([batch_size, 256], name='fc2_var')

    fc2_relu = tf.nn.relu(tf.matmul(fc1_var, W_fc2, name='fc2') + b_fc2, name='fc2_relu')
    fc2_loss = tf.reduce_sum(tf.square(tf.sub(fc2_var, fc2_relu)), name='fc2_loss') / (256*batch_size)
    fc2_reg = -tf.log(tf.reduce_sum(fc2_relu) / (256*batch_size) + 1)
    fc2_weight_loss = tf.reduce_sum(tf.square(W_fc2)) / (256*256)

with tf.name_scope("fc3"):
    W_fc3 = weight_variable([256, 32*32*3])
    b_fc3 = bias_variable([32*32*3])

    fc3_relu = tf.nn.relu(tf.matmul(fc2_var, W_fc3, name='fc3') + b_fc3, name='fc3_relu')
    fc3_loss = tf.reduce_sum(tf.square(tf.sub(x, fc3_relu)), name='fc3_loss') / (32*32*3*batch_size)
    fc3_weight_loss = tf.reduce_sum(tf.square(W_fc3)) / (256*32*28)

total_train_loss = fc1_loss * 10 + fc2_loss + fc3_loss # + #conv1_loss + conv2_loss
total_test_loss = fc1_loss * 100 + fc2_loss + fc3_loss

with tf.name_scope('summary'):
    tf.scalar_summary('fc1_loss', fc1_loss)
    tf.scalar_summary('fc2_loss', fc2_loss)
    tf.scalar_summary('fc3_loss', fc3_loss)
    tf.scalar_summary('fc1_weight_loss', fc1_weight_loss)
    tf.scalar_summary('fc2_weight_loss', fc2_weight_loss)
    tf.scalar_summary('fc3_weight_loss', fc3_weight_loss)
    tf.scalar_summary('fc1_reg', fc1_reg)
    tf.scalar_summary('fc2_reg', fc2_reg)
    #tf.scalar_summary('conv1_loss', conv1_loss)
    #tf.scalar_summary('conv2_loss', conv2_loss)
    tf.scalar_summary('total_loss', total_train_loss)
    tf.histogram_summary('fc1_weight_hist', W_fc1)
    tf.histogram_summary('fc2_weight_hist', W_fc2)
    tf.histogram_summary('fc3_weight_hist', W_fc3)

e_learning_rate = tf.placeholder(tf.float32, shape=[])
e_step = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(total_train_loss,
                                                                     var_list=[fc1_var, fc2_var, transform_var],
                                                                     name='E_optim')
m_learning_rate = tf.placeholder(tf.float32, shape=[])
m_step = tf.train.GradientDescentOptimizer(m_learning_rate).minimize(total_train_loss,
                                           var_list=[W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3],
                                           name='M_optim')

e_step_test = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(total_test_loss,
                                                                          var_list=[fc1_var, fc2_var, transform_var, y_weight_var],
                                                                          name='E_optim_test')
summary_op = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('log/generative', sess.graph)

y_vis = tf.placeholder(tf.float32, shape=[1, 12])
with tf.name_scope("visualization"):
    vis_fc1_relu = tf.nn.relu(tf.matmul(y_vis, W_fc1) + b_fc1)
    vis_fc2_relu = tf.nn.relu(tf.matmul(vis_fc1_relu, W_fc2) + b_fc2)
    image_out = tf.nn.relu(tf.matmul(vis_fc2_relu, W_fc3, name='fc2') + b_fc3, name='fc2_relu')

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if os.path.isfile('model.ckpt'):
    saver.restore(sess, "model.ckpt")
    print("Loading previous network")

e_lr = 2000
m_lr = 5
test_lr = 2000
e_step_size = 20
m_step_size = 10000
test_step_size = 100


def reinitialize():
    sess.run(fc1_var.initializer)
    sess.run(fc2_var.initializer)
    # sess.run(conv1_var.initializer)
    sess.run(y_weight_var.initializer)


def test_network():
    data, label = test_batch = dataset.next_test_batch(batch_size)
    reinitialize()
    test_lr = 200
    for e_iter in range(0, test_step_size):
        sess.run(e_step_test, feed_dict={x: data, y_ref: label,
                                         train_phase: [False]*batch_size, e_learning_rate: test_lr})
        test_lr *= 0.9
        # print(sess.run(fc1_var)[0, 0:10])
    print("Loss is " + str(sess.run(fc1_loss, feed_dict={x: data, y_ref: label,
                                            train_phase: [False]*batch_size})))
    # print(sess.run(y_weight_var))
    sess.run(y_weight_var.initializer)
    print("Loss is " + str(sess.run(fc1_loss, feed_dict={x: data, y_ref: label,
                                           train_phase: [False]*batch_size})))
    truth = np.argmax(test_batch[1], 1)
    pred = np.argmax(sess.run(y_weight_var), 1)
    print("Loss is " + str(sess.run(total_test_loss, feed_dict={x: data, y_ref: label,
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
        vis_result = dataset.convert_to_rgb(vis_result[:, :, 0])
        plt.imshow(vis_result)
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

transform_var_ub = np.zeros(transform_dim)
transform_var_lb = np.zeros(transform_dim)
for m_iter in range(m_step_size):
    batch_data, batch_label = dataset.next_batch(batch_size)
    if m_iter % 5 == 0:
        test_network()
    use_label = True
    reinitialize()
    e_lr = 1000
    for e_iter in range(0, e_step_size):
        sess.run(e_step, feed_dict={x: batch_data, y_ref: batch_label, train_phase: [use_label]*batch_size, e_learning_rate: e_lr})
        e_lr *= 0.9
        # print(sess.run(fc1_var)[0, 0:10])
    transform_value = sess.run(transform_var)
    for b in range(batch_size):
        for i in range(transform_dim):
            transform_var_ub[i] = max(transform_value[b, i], transform_var_ub[i])
            transform_var_lb[i] = min(transform_value[b, i], transform_var_lb[i])
    print(transform_var_ub, transform_var_lb)
    for e_iter in range(0, 2):
        sess.run(e_step, feed_dict={x: batch_data, y_ref: batch_label, train_phase: [use_label]*batch_size, e_learning_rate: e_lr})
        sess.run(m_step, feed_dict={x: batch_data, y_ref: batch_label, train_phase: [use_label]*batch_size, m_learning_rate: m_lr})
    summary_str, loss_result = sess.run([summary_op, total_train_loss], feed_dict={x: batch_data, y_ref: batch_label,
                                                  train_phase: [use_label]*batch_size})

    train_writer.add_summary(summary_str, m_iter)
    print("Iteration M: " + str(m_iter) + " with loss " + str(loss_result))
    train_writer.flush()
    if m_iter % 100 == 0:
        m_lr *= 0.95

    if m_iter % 20 == 0 and e_step_size < 30:
        e_step_size += 1
    if m_iter % 50 == 0 and m_iter != 0:
        save_path = saver.save(sess, "model.ckpt")
    if m_iter % 20 == 0 and m_iter != 0:
        visualize()





