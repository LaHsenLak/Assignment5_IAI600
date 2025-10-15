import sys
import time


assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)

#Poly
param_grid_poly = {
    'C': [0.1, 1, 10],
    'degree': [2, 3, 4],
    'coef0': [0, 1]
    }

svm_clf_poly = SVC(kernel= 'poly')
grid_search_poly = GridSearchCV(svm_clf_poly, param_grid_poly, cv=3, n_jobs=-1)
start_time_poly = time.time_ns()
grid_search_poly.fit(X_train, y_train)
end_time_poly = time.time_ns()
training_time_poly = end_time_poly - start_time_poly
y_predicted_poly = grid_search_poly.predict(X_test)
print("POLY")
print("F1-score: ", f1_score(y_test, y_predicted_poly), average='macro')
print("Accuracy score: ", accuracy_score(y_test, y_predicted_poly))
print("Precision score: ", precision_score(y_test, y_predicted_poly), average='macro')
print("Precision score: ", recall_score(y_test, y_predicted_poly), average='macro')
print("Time complexity: ", training_time_poly)
print()


#RBF
param_grid_rbf = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
    }

svm_clf_rbf = SVC(kernel= 'rbf')
grid_search_rbf = GridSearchCV(svm_clf_rbf, param_grid_rbf, cv=3, n_jobs=-1)
start_time_rbf = time.time_ns()
grid_search_rbf.fit(X_train, y_train)
end_time_rbf = time.time_ns()
training_time_rbf = end_time_rbf - start_time_rbf
y_predicted_rbf = grid_search_rbf.predict(X_test)
print("RBF")
print("F1-score: ", f1_score(y_test, y_predicted_rbf), average='macro')
print("Accuracy score: ", accuracy_score(y_test, y_predicted_rbf))
print("Precision score: ", precision_score(y_test, y_predicted_rbf), average='macro')
print("Precision score: ", recall_score(y_test, y_predicted_rbf), average='macro')
print("Time complexity: ", training_time_rbf)
print()


#Linear
param_grid_linear = {
        'C': [0.1, 1, 10]
     }

svm_clf_linear = SVC(kernel= 'linear')
grid_search_linear = GridSearchCV(svm_clf_linear, param_grid_linear, cv=3, n_jobs=-1)
start_time_linear = time.time_ns()
grid_search_linear.fit(X_train, y_train)
end_time_linear = time.time_ns()
training_time_linear = end_time_linear - start_time_linear
y_predicted_linear = grid_search_linear.predict(X_test)
print("Linear")
print("F1-score: ", f1_score(y_test, y_predicted_linear), average='macro')
print("Accuracy score: ", accuracy_score(y_test, y_predicted_linear))
print("Precision score: ", precision_score(y_test, y_predicted_linear), average='macro')
print("Precision score: ", recall_score(y_test, y_predicted_linear), average='macro')
print("Time complexity: ", training_time_linear)
