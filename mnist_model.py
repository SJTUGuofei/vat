import numpy as np
import tensorflow as tf

ftype = tf.float32
itype = tf.int32

num_classes = 10
hidden_dim = 1024

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=ftype)
def bias_variable(shape, scale=0.1):
    return tf.Variable(np.ones(shape)*scale, dtype=ftype)
def convolution(x, w, b):
    conv = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME") + b)
    return tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

class MLPModel:
    def __init__(self, keep_prob=None):
        self.keep_prob = keep_prob if keep_prob is None else tf.placeholder_with_default(1.0, [])
        self.weights = []
        self.biases = []
        self.w_fc1 = weight_variable([784, hidden_dim])
        self.b_fc1 = bias_variable([hidden_dim])
        self.weights.append(self.w_fc1)
        self.biases.append(self.b_fc1)
        self.w_fc2 = weight_variable([hidden_dim, num_classes])
        self.b_fc2 = bias_variable([num_classes])
        self.weights.append(self.w_fc2)
        self.biases.append(self.b_fc2)
    def __call__(self, input_op):
        h_fc1 = tf.nn.relu(tf.matmul(input_op, self.w_fc1) + self.b_fc1)
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        logits = tf.matmul(h_fc1, self.w_fc2) + self.b_fc2
        return logits


class CNNModel:
    def __init__(self, keep_prob=None):
        self.keep_prob = keep_prob if keep_prob is None else tf.placeholder_with_default(1.0, [])
        self.weights = []
        self.biases = []
        self.w_conv1 = weight_variable([5,5, 1,32])
        self.b_conv1 = bias_variable([32])
        self.weights.append(self.w_conv1)
        self.biases.append(self.b_conv1)
        self.w_conv2 = weight_variable([5,5, 32,64])
        self.b_conv2 = bias_variable([64])
        self.weights.append(self.w_conv2)
        self.biases.append(self.b_conv2)
        self.w_fc1 = weight_variable([7*7*64, hidden_dim])
        self.b_fc1 = bias_variable([hidden_dim])
        self.weights.append(self.w_fc1)
        self.biases.append(self.b_fc1)
        self.w_fc2 = weight_variable([hidden_dim, num_classes])
        self.b_fc2 = bias_variable([num_classes])
        self.weights.append(self.w_fc2)
        self.biases.append(self.b_fc2)
    def __call__(self, input_op, isTrain):
        input_image = tf.reshape(input_op, [-1, 28,28, 1])
        h_conv1 = convolution(input_image, self.w_conv1, self.b_conv1)
        h_conv2 = convolution(h_conv1, self.w_conv2, self.b_conv2)
        h_conv2_flat = tf.reshape(h_conv2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.w_fc1) + self.b_fc1)
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        logits = tf.matmul(h_fc1, self.w_fc2) + self.b_fc2
        return logits

