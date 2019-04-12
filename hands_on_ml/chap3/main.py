import __init__
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def print_samples(x_train):
    columns = 4
    rows = 5
    fig = plt.figure()
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(x_train[i-1], cmap='binary')
    plt.show()


def cross_validate(model, x_train, y_train):
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(x_train, y_train):
        clone_clf = clone(model)
        x_train_folds = x_train[train_index]
        y_train_folds = y_train[train_index]
        x_test_folds = x_train[test_index]
        y_test_folds = y_train[test_index]
        clone_clf.fit(x_train_folds, y_train_folds)
        y_pred = clone_clf.predict(x_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        print(n_correct / len(y_pred))


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1),dtype=bool)


def print_f1_score(true_labels, pred_labels):
    print("pred false, pred true")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    f1 = 2 * precision*recall / (precision + recall)
    print("precision = TP/(TP+FP)")
    print(precision)
    print("Recall = TP/(TP + FN)")
    print(recall)
    print("F1 = 2 * P*R / (P + R)")
    print(f1)


def plot_precision_recall(precision, recalls, thresholds):
    plt.plot(thresholds, precision[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0,1])


def run_not_fives(x_train, y_train):
    y_train_5 = (y_train == 5)
    rf_clf = RandomForestClassifier(random_state=42)
    y_scores_prob = cross_val_predict(rf_clf, x_train, y_train_5, cv=3, method='predict_proba')
    y_scores_rf = y_scores_prob[:, 1]
    y_preds = (y_scores_rf > 0.5)
    print_f1_score(y_train_5, y_preds)


def run_ovo_sgd(x_train, y_train):
    #ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
    ovo_clf = SGDClassifier(random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
    print(cross_val_score(ovo_clf, x_train_scaled, y_train, cv=3, scoring='accuracy'))
    sgd_clf = SGDClassifier(random_state=42)
    print(cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring='accuracy'))


def run_rf_multiclass(x_train, y_train):
    rf_clf = RandomForestClassifier()
    print(cross_val_score(rf_clf, x_train, y_train, cv=3, scoring='accuracy'))
    rf_clf.fit(x_train, y_train)
    print(rf_clf.predict([x_train[0]]))
    print(rf_clf.predict_proba([x_train[0]]))


def run_confusion_matrix(x_train_reshaped, y_train):
    sgd_clf = SGDClassifier()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_reshaped)
    y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)
    plt.matshow(conf_mx)
    plt.show()
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx/row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    print(norm_conf_mx)
    plt.matshow(norm_conf_mx)
    plt.show()


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_reshaped = x_train.reshape(-1, 28 * 28)
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train %2 == 1)
