import __init__
import tensorflow as tf
import random
import numpy as np
import src.load_data.loader as data_loader
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_data():
    url = data_loader.dataset_path("housing", "housing.csv")
    housing = pd.read_csv(url)
    housing = housing.dropna()
    housing_train = housing.drop(["ocean_proximity", "median_house_value"], axis=1)
    housing_target = housing["median_house_value"]
    print("Housing data:")
    print(housing_train.head())
    x_train = housing_train.values
    m, n = x_train.shape
    x_train_plus_bias = np.c_[np.ones((m,1)), x_train]
    y_train = housing_target.values
    return x_train_plus_bias, y_train


def run_linear_regression(X_train, y_train):
    # create linear regression graph
    X = tf.constant(X_train, dtype=tf.float32, name='X')
    y = tf.constant(y_train.reshape(-1,1), dtype=tf.float32, name='y')
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    with tf.Session() as sess:
        print(theta.eval())


def run_sgd_regression(X_train, y_train):
    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    n_epochs = 1000
    learning_rate = 0.01
    m, n = X_train.shape
    X = tf.constant(X_train_scaled, dtype=tf.float32, name='X')
    y = tf.constant(y_train, dtype=tf.float32, name='y')
    theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(error.eval())




if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    pd.set_option('display.max_columns', None)
    x_housing, y_housing = get_data()
    #run_linear_regression(x_housing, y_housing)
    run_sgd_regression(x_housing, y_housing)



