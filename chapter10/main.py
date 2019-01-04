import os
import sys
import __init__
import tensorflow as tf
import src.load_data.loader as data_loader
import numpy as np


def layer(X, n_units, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt((n_inputs + n_units)*1.0)
        init = tf.truncated_normal((n_inputs, n_units), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_units]), name='bias')
        z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(z)
        else:
            return z

def create_graph():
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.float64, shape=(None), name='y')
    layer(X, n_hidden1, "hi")


if __name__ == "__main__":
    image_path = data_loader.dataset_path("mnist", "mnist.npz")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=image_path)
    n_features = 28*28
    x_train = x_train.reshape(-1, n_features)
    x_test = x_test.reshape(-1, n_features)
    create_graph()