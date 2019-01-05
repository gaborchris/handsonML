import __init__
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_image

import matplotlib.pyplot as plt
import matplotlib.image as mimage

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
    plt.imshow(output[0,:,:,1], cmap="gray")
    plt.show()
    plt.imshow(output[1,:,:,0], cmap="gray")
    plt.show()
    plt.imshow(output[1,:,:,1], cmap="gray")
    plt.show()










if __name__ == "__main__":
    test_filter()



