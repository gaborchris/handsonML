import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
if __name__ == "__main__":
    mnist = input_data.read_data_sets("/tmp/data/")

    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    #file_graph = tf.summary.FileWriter("./tf_log", tf.get_default_graph())
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    X_test = mnist.test.images
    y_test = mnist.test.labels

    with tf.Session() as sess:
        saver.restore(sess, "./tf_log/mymodel_final.ckpt")
        Z = logits.eval(feed_dict={X: X_test})
        for i in range(10):
            print(np.argmax(Z[50+i,:]))
            some_digit = X_test[50+i]
            some_digit_image = some_digit.reshape(28, 28)
            plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
            plt.show()