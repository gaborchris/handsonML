import __init__
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
import src.load_data.loader as data_loader
import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
#warnings.warn = warn
def generate_line_data(xrange, x0, x1):
    X = xrange * np.random.rand(xrange*50, 1)
    y = x0 + (x1 * X) + np.random.randn(xrange*50, 1)
    X_b = np.c_[np.ones((xrange*50, 1)), X]
    return X_b, y
def fit_lin_reg(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
def plot_best_fit(x_range, lin_params, X_sca, y_sca):
    X = np.c_[np.ones((2, 1)), np.array([0, x_range])]
    y_predict = X.dot(lin_params)
    plt.scatter(X_sca[:, 1], y_sca)
    plt.plot(X[:, 1], y_predict)
    plt.axis([0, x_range, 0, (y_predict[-1] + 5)])
    plt.show()
def lin_reg_example():
    x_range = 5
    x, y = generate_line_data(x_range, 1, 3)
    best_theta = fit_lin_reg(x, y)
    print(best_theta)
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    theta = [[lin_reg.intercept_[0]], [lin_reg.coef_[0,1]]]
    print(theta)
    plot_best_fit(x_range, theta, x, y)
def grad_descent_example():
    x_range = 5
    X, y = generate_line_data(x_range, 1, 3)
    eta = 0.1
    n_iters = 1000
    m = x_range*50
    theta = np.random.randn(2, 1)
    print(theta)
    for iteration in range(n_iters):
        gradients = 2 / m *X.T.dot(X.dot(theta) - y)
        theta = theta - eta*gradients
    print(theta)
    plot_best_fit(x_range, theta, X, y)
def learning_schedule(t, t0, t1):
    return t0 / (t + t1)
def sgd_example():
    x_range = 2
    m = x_range * 50
    X, y = generate_line_data(x_range, 4, 2)
    n_epochs = 50
    t0, t1 = 5, 50
    theta = np.random.randn(2, 1)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index: random_index+1]
            yi = y[random_index: random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch*m + i, t0, t1)
            theta = theta - eta * gradients
        print(theta)
def built_in_regressor():
    sgd_reg = SGDRegressor(penalty=None, eta0=0.01)
    x_range = 20
    m = x_range * 50
    X, y = generate_line_data(x_range, 10, 5)
    sgd_reg.fit(X,y.ravel())
    print(sgd_reg.intercept_, sgd_reg.coef_)
def generate_poly_data(degree, xrange, m, x0, x1, x2):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X = xrange * np.random.rand(m, 1) - xrange/2.0
    y = x2 * X**2 + x1 * X + x0 + np.random.randn(m, 1)/2
    X_poly = poly_features.fit_transform(X)
    X_b = np.c_[np.ones((m, 1)), X_poly]
    return X_b, y
def poly_regression():
    X, y = generate_poly_data(6, 100, 6, 1, 2)
    poly_reg = SGDRegressor(max_iter=100000, eta0=0.0001)
    poly_reg.fit(X, y.ravel())
    print(poly_reg.intercept_, poly_reg.coef_)
    xline = np.linspace(-3, 3, 100)
    x_test = np.c_[xline, xline**2]
    print(x_test[0])
    y_pred = poly_reg.predict(x_test)
    plt.scatter(X[:, 0], y)
    plt.plot(xline, y_pred, color='red')
    plt.show()
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)
    train_errors, val_errors = [],[]
    for m in range(1, len(X_train)):
        print("m=", m)
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot( np.sqrt(train_errors), "r-+", label="train")
    plt.plot( np.sqrt(val_errors), "b-", label="val")
    plt.legend()
    plt.show()
def get_even_spaced(degrees, start, end):
    poly_features = PolynomialFeatures(degree=degrees, include_bias=False)
    X = np.linspace(start, end, 300)
    X = X.reshape(-1, 1)
    X_poly = poly_features.fit_transform(X)
    X_b = np.c_[np.ones((300, 1)), X_poly]
    return X_b
def overfitting():
    for degrees in range(1, 100, 2):
        X, y = generate_poly_data(degrees, 6, 100, 5, 1, 2)
        #sgd_reg = SGDRegressor(max_iter=1000, eta0=0.0001)
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        xline = get_even_spaced(degrees)
        ypred = lin_reg.predict(xline)
        plt.scatter(X[:,1], y)
        plt.plot(xline[:,1], ypred, color='red')
        plt.ylim(0,30)
        plt.show()

def logit(x):
    return 1/(1 + 2.7182**(-x))


if __name__ == "__main__":
    #url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv"
    #data_loader.fetch_data(url, "iris", "iris.csv")
    flowertype = "virginica"
    data_path = data_loader.dataset_path("iris", "iris.csv")
    df = pd.read_csv(data_path)
    X = df.values[:,3:4]
    y = (df.values[:,4] == flowertype).astype(np.int)
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    xnew = np.linspace(0, 3, 100).reshape(-1,1)
    y_proba = log_reg.predict_proba(xnew)
    plt.plot(xnew, y_proba[:, 1], label=flowertype)
    plt.plot(xnew, y_proba[:, 0], label="not " + flowertype)
    plt.scatter(X[y==True], np.zeros(X[y==True].shape), label=flowertype + " instance")
    plt.scatter(X[y == False], np.ones(X[y == False].shape), label="non " + flowertype + " instance")
    plt.legend()
    plt.show()

    X = df.values[:,2:4]
    y = df["species"].astype('category').cat.codes.values
    #df_with_dummies = pd.get_dummies(columns=cols_to_transform)
    softmax_reg = LogisticRegression(multi_class="multinomial", solver='lbfgs', C=10)
    softmax_reg.fit(X, y)
    xgrid, ygrid = np.meshgrid(np.linspace(0, 8, 100), np.linspace(0, 4, 100))
    Z = softmax_reg.predict(np.c_[xgrid.ravel(), ygrid.ravel()])
    Z = Z.reshape(xgrid.shape)

    plt.contourf(xgrid, ygrid, Z)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='1',color='red')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='2', color='orange')
    plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], label='3', color='cyan')
    plt.legend()
    plt.show()












