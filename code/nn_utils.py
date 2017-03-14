import tensorflow as tf
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01, dtype = tf.float32)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape = shape, dtype = tf.float32)
	return tf.Variable(initial)

def conv2d(x, W, stride):
	x = tf.cast(x, tf.float32)
	return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
	x = tf.cast(x, tf.float32)
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")