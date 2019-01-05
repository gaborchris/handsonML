import os
import sys
import __init__
import tensorflow as tf
import src.load_data.loader as data_loader
import numpy as np
from functools import partial
from datetime import datetime

def leaky_relu(z, name=None):
    return tf.maximum(0.2*z, z, name=name)


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
    initial_learning_rate = 0.1
    decay_steps = 100000
    decay_rate=0.1
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

    n_inputs = 28*28
    n_hidden1 = 500
    n_hidden2 = 200
    n_hidden3 = 100
    n_outputs = 10
    reg_scale = 0.001
    dropout_rate = 0.1


    training = tf.placeholder_with_default(False, shape=(), name="training")
    my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    X_drop = tf.layers.dropout(X, dropout_rate, training=training)

    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer()

        hidden1 = tf.layers.dense(X_drop, n_hidden1, name="hidden1", kernel_regularizer=tf.contrib.layers.l1_regularizer(reg_scale))
        bn1 = my_batch_norm_layer(hidden1)
        bn1_act = tf.nn.elu(bn1)
        hidden1_drop = tf.layers.dropout(bn1_act, dropout_rate, training=training)

        hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2", kernel_initializer=he_init, kernel_regularizer=tf.contrib.layers.l1_regularizer(reg_scale))
        bn2 = my_batch_norm_layer(hidden2)
        bn2_act = tf.nn.elu(bn2)
        hidden2_drop = tf.layers.dropout(bn2_act, dropout_rate, training=training)

        hidden3 = tf.layers.dense(hidden2_drop, n_hidden3, name="hidden3", kernel_initializer=he_init, kernel_regularizer=tf.contrib.layers.l1_regularizer(reg_scale))
        bn3 = my_batch_norm_layer(hidden3)
        bn3_act = tf.nn.elu(bn3)

        logits_before = tf.layers.dense(bn3_act, n_outputs, name="outputs")
        logits = my_batch_norm_layer(logits_before)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    with tf.name_scope("train"):
        #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        training_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        #training_op = optimizer.minimize(loss)
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    for op in (X, y, accuracy, training_op):
        tf.add_to_collection("my_important_ops", op)

    n_epochs = 200
    batch_size = 500

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            start_time = datetime.now()
            for batch_index in range(int(y_train.shape[0]/batch_size)):
                b_start, b_end = batch_size*batch_index, batch_size*batch_index + batch_size
                X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]
                sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
            end_time = datetime.now()
            print("Elapsed: ", (end_time-start_time).total_seconds())
            saver.save(sess, "./my_model.ckpt")
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
