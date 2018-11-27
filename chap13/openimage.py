import tensorflow as tf
import numpy as np
from sklearn .datasets import load_sample_image
import matplotlib.pyplot as plt
#from PIL import Image

if __name__ == "__main__":
    china = load_sample_image("china.jpg")
    flower = load_sample_image("flower.jpg")
    dataset = np.array([china, flower], dtype=np.float32)
    batch_size, height, width, channels = dataset.shape
    print(dataset.shape)

    filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
    filters[:, 3, :, 0] = 1
    filters[3, :, :, 1] = 1

    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    #convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")
    max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    with tf.Session() as sess:
        #output = sess.run(convolution, feed_dict={X: dataset})
        output = sess.run(max_pool, feed_dict={X: dataset})


    print(output.shape)
    plt.imshow(china)
    plt.show()
    plt.imshow(output[0])
    plt.imshow(output[0, :, :, 0], cmap="gray")
    plt.show()
    plt.imshow(output[0, :, :, 1], cmap="gray")
    plt.show()
    plt.imshow(flower)
    plt.show()
    plt.imshow(output[1, :, :, 0], cmap="gray")
    plt.show()
    plt.imshow(output[1, :, :, 1], cmap="gray")
    plt.show()
