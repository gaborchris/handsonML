import os
import sys
import __init__
import tensorflow as tf
import src.load_data.loader as data_loader
import numpy as np

def leaky_relu(z, name=None):
    return tf.maximum(0.01*z, z, name=name)


def layer(X, n_units, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt((n_inputs + n_units)*1.0)
        print(stddev)
        init = tf.truncated_normal((n_inputs, n_units), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_units]), name='bias')
        z = tf.matmul(X, W) + b
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(W.eval())
        if activation is not None:
            return activation(z)
        else:
            return z

def create_and_run_graph(X_train, y_train, X_val, y_val):
    learning_rate = 0.01
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        #hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", kernel_initializer=he_init, activation=tf.nn.relu)
        #hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", kernel_initializer=he_init, activation=tf.nn.relu)
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.elu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.elu)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

        '''
        hidden1 = layer(X, n_hidden1, "hidden1", activation=tf.nn.relu)
        hidden2 = layer(hidden1, n_hidden2, "hidden2", activation=tf.nn.relu)
        logits = layer(hidden2, n_outputs, name="outputs")
        '''
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 40
    batch_size = 50

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(int(y_train.shape[0]/batch_size)):
                b_start, b_end = batch_size*batch_index, batch_size*batch_index + batch_size
                X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
            print("Epoch: ", epoch, "training: ", acc_train, "val_acc", acc_val)








if __name__ == "__main__":
    image_path = data_loader.dataset_path("mnist", "mnist.npz")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=image_path)
    n_features = 28*28
    x_train = x_train.reshape(-1, n_features)
    x_train = x_train/255
    x_test = x_test.reshape(-1, n_features)
    x_test = x_test/255
    print(y_train.shape)
    create_and_run_graph(x_train, y_train, x_test, y_test)
