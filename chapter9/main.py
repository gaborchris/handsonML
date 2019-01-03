import __init__
import tensorflow as tf
import random
import numpy as np
import src.load_data.loader as data_loader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def get_data(scaled=False):
    url = data_loader.dataset_path("housing", "housing.csv")
    housing = pd.read_csv(url)
    housing = housing.dropna()
    housing_train = housing.drop(["ocean_proximity", "median_house_value"], axis=1)
    housing_target = housing["median_house_value"]
    print("Housing data:")
    print(housing_train.head())
    x_train = housing_train.values
    m, n = x_train.shape
    if scaled:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
    x_train_plus_bias = np.c_[np.ones((m, 1)), x_train]
    y_train = housing_target.values.reshape(-1,1)
    return x_train_plus_bias, y_train



def run_linear_regression(X_train, y_train):
    X = tf.constant(X_train, dtype=tf.float32, name='X')
    y = tf.constant(y_train.reshape(-1,1), dtype=tf.float32, name='y')
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    y_pred = tf.matmul(X, theta, name="preds")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

    with tf.Session() as sess:
        print("RMS error")
        print(np.sqrt(mse.eval()))
        print(theta.eval())


def run_gradient_batch_regression(X_train, y_train):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}".format(root_logdir, now)

    m, n = X_train.shape
    n_epochs = 2000
    learning_rate = 0.01
    X = tf.constant(X_train, dtype=tf.float32, name='X')
    y = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32, name='y')
    theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), dtype=tf.float32, name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch ", epoch, "RMSE=", np.sqrt(mse.eval()))
                saver.save(sess, "./my_model.ckpt")
            sess.run(training_op)
            summary_str = mse_summary.eval()
            file_writer.add_summary(summary_str, epoch)
        print(theta.eval())
        file_writer.close()


def load_model_and_pred(X_train, y_train):
    saver = tf.train.import_meta_graph("./my_model.ckpt.meta")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "./my_model.ckpt")
        theta = tf.get_default_graph().get_tensor_by_name('theta:0')
        print(theta.eval())


def fetch_batch(X_train, y_train, batch_index, batch_size):
    X = X_train[batch_size*batch_index: batch_size*batch_index + batch_size]
    y = y_train[batch_size*batch_index: batch_size*batch_index + batch_size]
    return X, y


def mini_batch_regression(X_train, y_train):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}".format(root_logdir, now)

    m, n = X_train.shape
    n_epochs = 100
    learning_rate = 0.001
    batch_size = 100
    n_batches = int(np.ceil(m/batch_size))
    X = tf.placeholder(tf.float32, shape=(None, n), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), dtype=tf.float32, name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    with tf.name_scope("loss") as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name='mse')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(mse)

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch in range(n_batches):
                x_batch, y_batch = fetch_batch(X_train, y_train, batch, batch_size)
                sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
            if epoch % 10 == 0:
                print("Epoch ", epoch, "RMSE=", np.sqrt(mse.eval(feed_dict={X: X_train, y: y_train})))
                summary_str = mse_summary.eval(feed_dict={X: X_train, y: y_train})
                file_writer.add_summary(summary_str, epoch)
        print(theta.eval())

def relu(X):
    with tf.variable_scope("relu", reuse=tf.AUTO_REUSE):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        #with tf.Session() as sess:
        #    sess.run(tf.global_variables_initializer())
        #    print(threshold.eval())
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, threshold, name='relu')


def create_relu_compact():
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}".format(root_logdir, now)

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')

    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.1))
    relus = [relu(X) for i in range(5)]
    output = tf.add_n(relus, name='output')
    init = tf.global_variables_initializer()

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        #print(threshold.eval())
        file_writer.add_graph(tf.get_default_graph())



def create_relu():
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}".format(root_logdir, now)

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')

    w1 = tf.Variable(tf.random_normal((n_features, 1)), name='weights1')
    w2 = tf.Variable(tf.random_normal((n_features, 1)), name='weights2')
    b1 = tf.Variable(0.0, name='bias1')
    b2 = tf.Variable(0.0, name='bias2')

    z1 = tf.add(tf.matmul(X, w1), b1, name='z1')
    z2 = tf.add(tf.matmul(X, w2), b2, name='z2')

    relu1 = tf.maximum(z1, 0.0, name='relu1')
    relu2 = tf.maximum(z2, 0.0, name='relu2')

    output = tf.add(relu1, relu2, name='output')

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        file_writer.add_graph(tf.get_default_graph())


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    pd.set_option('display.max_columns', None)
    x_housing, y_housing = get_data(scaled=True)
    #mini_batch_regression(x_housing, y_housing)
    #run_gradient_batch_regression(x_housing, y_housing)
    #load_model_and_pred(x_housing, y_housing)
    #create_relu()
    create_relu_compact()







