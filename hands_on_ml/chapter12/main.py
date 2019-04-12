import __init__
import tensorflow as tf
import numpy as np
import src.load_data.loader as data_loader
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial

def test_filter():
    china = load_sample_image("china.jpg")
    flower = load_sample_image("flower.jpg")
    dataset = np.array([china, flower], dtype=np.float32)
    batch_size, height, width, channels = dataset.shape

    filters = np.zeros(shape=(7,7, channels, 2), dtype=np.float32)
    filters[:, 3, :, 0] = 1 # vertical line
    filters[3, :, :, 1] = 1 # horizontal line

    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    #conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2], padding='SAME')
    conv = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding='SAME')
    max_pool = tf.nn.avg_pool(conv, ksize=[1,4,4,1], strides=[1,4,4,1], padding="VALID")
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(max_pool, feed_dict={X: dataset})

    print(output.shape)
    plt.imshow(output[0,:,:,0], cmap="gray")
    plt.show()

def mnist_cnn(X_train, y_train, X_val, y_val):
    learning_rate = 0.1
    n_filters1 = 20
    n_filters2 = 40
    n_filters3 = 80
    n_dense1 = 200
    n_outputs = 10

    training = tf.placeholder_with_default(False, shape=(), name="training")
    batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

    X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    with tf.name_scope("cnn"):
        filters1 = np.random.randn(5,5,1, n_filters1)
        conv1 = tf.nn.conv2d(X, filters1, strides=[1,1,1,1], padding='SAME')
        bn1 = batch_norm_layer(conv1)
        bn1_act = tf.nn.elu(bn1)
        max_pool1 = tf.nn.max_pool(bn1_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

        filters2 = np.random.randn(3,3,n_filters1, n_filters2)
        conv2 = tf.nn.conv2d(max_pool1, filters2, strides=[1,1,1,1], padding='SAME')
        bn2 = batch_norm_layer(conv2)
        bn2_act = tf.nn.elu(bn2)
        max_pool2 = tf.nn.max_pool(bn2_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

        filters3 = np.random.randn(3,3,n_filters2, n_filters3)
        conv3 = tf.nn.conv2d(max_pool2, filters3, strides=[1,1,1,1], padding='VALID')
        bn3 = batch_norm_layer(conv3)
        bn3_act = tf.nn.elu(bn3)

        flat = tf.layers.flatten(bn3_act)
        fc1 = tf.layers.dense(flat, n_dense1, kernel_regularizer=tf.contrib.layers.l1_regularizer(0.001))

        logits = tf.layers.dense(fc1, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    init = tf.global_variables_initializer()

    n_epochs = 200
    batch_size = 500

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            start_time = datetime.now()
            for batch_index in range(int(y_train.shape[0]/batch_size)):
                b_start, b_end = batch_size*batch_index, batch_size*batch_index + batch_size
                X_batch, y_batch = X_train[b_start: b_end, :, :, :], y_train[b_start: b_end]
                sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
            end_time = datetime.now()
            print("Elapsed: ", (end_time-start_time).total_seconds())
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
            print("Epoch: ", epoch, "training: ", acc_train, "val_acc", acc_val)


if __name__ == "__main__":
    image_path = data_loader.dataset_path("mnist", "mnist.npz")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=image_path)
    x_train = x_train/255
    x_test = x_test/255
    print(x_train.shape)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    mnist_cnn(x_train, y_train, x_test, y_test)







