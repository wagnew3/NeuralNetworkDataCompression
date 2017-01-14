# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
##from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

# read in data
x_train = genfromtxt('/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000TrainIn.csv',delimiter=',')  # Training input
y_train_onehot = genfromtxt('/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000TrainOut.csv',delimiter=',')      # Training output
x_test = genfromtxt('/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000ValIn.csv',delimiter=',')  # Testing input
y_test_onehot = genfromtxt('/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000ValOut.csv',delimiter=',')  # Testing output

print(x_test[0])
print(y_test_onehot[0])

##flags = tf.app.flags
##FLAGS = flags.FLAGS
##flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

##mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

# Create the model
##x = tf.placeholder(tf.float32, [None, 784])
##W = tf.Variable(tf.zeros([784, 10]))
##b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 49])
W = tf.Variable(tf.zeros([49, 15]))
b = tf.Variable(tf.zeros([15]))
y = tf.nn.relu(tf.matmul(x, W) + b)

# Define loss and optimizer
##y_ = tf.placeholder(tf.float32, [None, 10])
y_ = tf.placeholder(tf.float32, [None, 15])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
for i in range(1000):
  ##batch_xs, batch_ys = mnist.train.next_batch(100)
  ##train_step.run({x: batch_xs, y_: batch_ys})
  train_step.run({x: x_train, y_: y_train_onehot})

# Test trained model
  correct_prediction = tf.reduce_sum(tf.abs(y-y_))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  ##print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
  print(accuracy.eval({x: x_test, y_: y_test_onehot}))
