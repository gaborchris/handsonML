import tensorflow as tf
import matplotlib.pyplot
mnist = tf.keras.datasets.mnist


def load():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    return (x_train, y_train), (x_test, y_test)


if __name__== "__main__":
    (x_train, y_train), (x_test, y_test) = load()
    print(x_train[0].shape)
    print(x_train.shape, x_test.shape, y_train.shape)
    matplotlib.pyplot.imshow(x_train[0])
    print(x_train[0])
    matplotlib.pyplot.show()