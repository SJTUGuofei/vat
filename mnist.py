import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from vat import VAT
import mnist_model

do_VAT = True

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
max_epoch = 100
keep_prob_during_training = 1.0

x = tf.placeholder(mnist_model.ftype, [None, 784])
labels = tf.placeholder(mnist_model.itype, [None, 10])
keep_prob = tf.placeholder_with_default(1.0, [])

#optimizer = tf.train.GradientDescentOptimizer(0.001)
optimizer = tf.train.AdamOptimizer()

network = mnist_model.MLPModel(keep_prob=keep_prob)
#network = mnist_model.CNNModel(keep_prob=keep_prob)

logits = network(x)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
vat_cross_entropy, vat_perturbation = VAT(x, network)
predictions = tf.argmax(logits, axis=1)
accuracy, update_accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions)

def cross_entropy_and_accuracy(feed_dict):
    global sess, cross_entropy, accuracy, update_accuracy
    sess.run(tf.local_variables_initializer())
    loss, _ = sess.run([cross_entropy, update_accuracy], feed_dict=feed_dict)
    acc = sess.run(accuracy)
    return loss, acc

train_loss = cross_entropy
if do_VAT:
    train_loss += vat_cross_entropy
train_op = optimizer.minimize(train_loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

loss, acc = cross_entropy_and_accuracy({x:mnist.validation.images, labels:mnist.validation.labels})
print("initial    loss: %f  accuracy: %f" % (loss, acc))
best_accuracy = acc
best_epoch = -1
for epoch in range(max_epoch):
    for i in range(0, 55000, 100):
        minibatch_x, minibatch_labels = mnist.train.next_batch(batch_size)
        fd = {x:minibatch_x, labels:minibatch_labels, keep_prob:keep_prob_during_training}
        sess.run(train_op, feed_dict=fd)

    loss, acc = cross_entropy_and_accuracy({x:mnist.validation.images, labels:mnist.validation.labels})
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch
    print("epoch: %2d  loss: %f  accuracy: %f" % (epoch, loss, acc))

print("best epoch: %2d  accuracy: %f" % (best_epoch, best_accuracy))








