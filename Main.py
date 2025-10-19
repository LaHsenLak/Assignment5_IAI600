import sys
import time


assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import pandas as pd
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier





mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)

#Poly grid
param_grid_poly = {
    'C': [0.1, 1, 10],
    'degree': [2, 3, 4],
    'coef0': [0, 1]
    }

svm_clf_poly = SVC(kernel= 'poly')
grid_search_poly = GridSearchCV(svm_clf_poly, param_grid_poly, cv=3, n_jobs=-1)
grid_search_poly.fit(X_train, y_train)
print("POLY GRID")
print("Best params", grid_search_poly.best_params_)
print()

#RBF grid
param_grid_rbf = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
    }

svm_clf_rbf = SVC(kernel= 'rbf')
grid_search_rbf = GridSearchCV(svm_clf_rbf, param_grid_rbf, cv=3, n_jobs=-1)
grid_search_rbf.fit(X_train, y_train)

print("RBF GRID")
print("Best params", grid_search_rbf.best_params_)
print()


# Poly kernel best model
svm_clf_poly = grid_search_poly.best_estimator_              # grid_search_poly.best_estimator_ = SVC(kernel= 'poly', C=10, coef0 = 1, degree = 4) 
svm_clf_poly.fit(X_train, y_train)
y_predicted_poly = svm_clf_poly.predict(X_test)

f1_score_poly = f1_score(y_test, y_predicted_poly, average='macro')
accuracy_score_poly = accuracy_score(y_test, y_predicted_poly)
precision_score_poly = precision_score(y_test, y_predicted_poly, average='macro')
recall_score_poly = recall_score(y_test, y_predicted_poly, average='macro')
print("POLY MODEL")
print("F1-score: ", f1_score_poly)
print("Accuracy score: ", accuracy_score_poly)
print("Precision score: ", precision_score_poly)
print("Recall score: ", recall_score_poly)
print()


# RBF kernel best model
print("RBF MODEL")
svm_clf_rbf = param_grid_rbf.best_estimator_          # param_grid_rbf.best_estimator_ = SVC(kernel= 'rbf', C=10, gamma='scale')
startTime = time.time()
svm_clf_rbf.fit(X_train, y_train)
endTime = time.time()
y_predicted_rbf = svm_clf_rbf.predict(X_test)

f1_score_rbf = f1_score(y_test, y_predicted_rbf, average='macro')
accuracy_score_rbf = accuracy_score(y_test, y_predicted_rbf)
precision_score_rbf = precision_score(y_test, y_predicted_rbf, average='macro')
recall_score_rbf = recall_score(y_test, y_predicted_rbf, average='macro')
trainingTimeRBF = endTime - startTime
print("F1-score: ", f1_score_rbf)
print("Accuracy score: ", accuracy_score_rbf)
print("Precision score: ", precision_score_rbf)
print("Recall score: ", recall_score_rbf)
print("Training time: ", trainingTimeRBF)
print()

# Linear model
print("LINEAR MODEL")
svm_clf_linear = SVC(kernel= 'linear', C=10)
svm_clf_linear.fit(X_train, y_train)
y_predicted_Linear = svm_clf_linear.predict(X_test)

f1_score_linear = f1_score(y_test, y_predicted_Linear, average='macro')
accuracy_score_linear = accuracy_score(y_test, y_predicted_Linear)
precision_score_linear = precision_score(y_test, y_predicted_Linear, average='macro')
recall_score_linear = recall_score(y_test, y_predicted_Linear, average='macro')
print("F1-score: ", f1_score_linear)
print("Accuracy score: ", accuracy_score_linear)
print("Precision score: ", precision_score_linear)
print("Recall score: ", recall_score_linear)
print()


# Collect all metrics into a dictionary
results = {
    'Linear': {
        'Accuracy': accuracy_score_linear,
        'Precision': precision_score_linear,            
        'Recall': recall_score_linear,                
        'F1-score': f1_score_linear                     
    },
    'Polynomial': {
        'Accuracy': accuracy_score_poly, 
        'Precision': precision_score_poly,  
        'Recall': recall_score_poly,   
        'F1-score': f1_score_poly        
    },
    'RBF': {
        'Accuracy': accuracy_score_rbf,
        'Precision': precision_score_rbf,
        'Recall': recall_score_rbf,
        'F1-score': f1_score_rbf
    }
}

df_results = pd.DataFrame(results).T

# Plot bar chart for kernels comparison
ax = df_results[["Accuracy","Precision","Recall","F1-score"]].plot(kind="bar", figsize=(12,6))
plt.title("SVM Kernels — MNIST")
plt.ylabel("Score")
plt.ylim(0.975, 0.985)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=0)
for c in ax.containers:
    ax.bar_label(c, fmt="%.4f", padding=2, rotation=90, label_type="edge", fontsize=8)
plt.tight_layout()
plt.show()



####################################

# KNN 
print("KNN")
knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance')  
startTime = time.time()
knn_clf.fit(X_train, y_train)
endTime = time.time()
y_predicted_KNN = knn_clf.predict(X_test)

f1_score_KNN = f1_score(y_test, y_predicted_KNN, average='macro')
accuracy_score_KNN = accuracy_score(y_test, y_predicted_KNN)
precision_score_KNN = precision_score(y_test, y_predicted_KNN, average='macro')
recall_score_KNN = recall_score(y_test, y_predicted_KNN, average='macro')
trainingTimeKNN = endTime - startTime

# SGD
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
startTime = time.time()
sgd_clf.fit(X_train, y_train)
endTime = time.time()
y_predicted_sgd = sgd_clf.predict(X_test)

f1_score_sgd = f1_score(y_test, y_predicted_sgd, average='macro')
accuracy_score_sgd = accuracy_score(y_test, y_predicted_sgd)
precision_score_sgd = precision_score(y_test, y_predicted_sgd, average='macro')
recall_score_sgd = recall_score(y_test, y_predicted_sgd, average='macro')
trainingTimeSGD = endTime - startTime

# RND FOREST
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
startTime = time.time() 
forest_clf.fit(X_train, y_train)
endTime = time.time()
y_predicted_forest = forest_clf.predict(X_test)

f1_score_forest = f1_score(y_test, y_predicted_forest, average='macro')
accuracy_score_forest = accuracy_score(y_test, y_predicted_forest)
precision_score_forest = precision_score(y_test, y_predicted_forest, average='macro')
recall_score_forest = recall_score(y_test, y_predicted_forest, average='macro')
trainingTimeRND = endTime - startTime


# RBF vs KNN / SGD / Random Forest: metrics comparison
compare = {
    "SVM-RBF": {
        "Accuracy": accuracy_score_rbf,
        "Precision": precision_score_rbf,
        "Recall": recall_score_rbf,
        "F1-score": f1_score_rbf
    },
    "KNN": {
        "Accuracy": accuracy_score_KNN,
        "Precision": precision_score_KNN,
        "Recall": recall_score_KNN,
        "F1-score": f1_score_KNN
    },
    "SGD": {
        "Accuracy": accuracy_score_sgd,
        "Precision": precision_score_sgd,
        "Recall": recall_score_sgd,
        "F1-score": f1_score_sgd
    },
    "RandomForest": {
        "Accuracy": accuracy_score_forest,
        "Precision": precision_score_forest,
        "Recall": recall_score_forest,
        "F1-score": f1_score_forest
    }
}

df_compare = pd.DataFrame(compare).T

# Comparison between models with bar chart for metrics
ax = df_compare[["Accuracy","Precision","Recall","F1-score"]].plot(kind="bar", figsize=(12,6))
plt.title("RBF SVM vs KNN / SGD / Random Forest — MNIST (Test Set)")
plt.ylabel("Score")
plt.ylim(0.85, 1) 
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=0)
for c in ax.containers:
    ax.bar_label(c, fmt="%.4f", padding=2, rotation=90, label_type="edge", fontsize=8)
plt.tight_layout()
plt.show()

# Training time comparison
train_times_ms = pd.Series({
    "SVM-RBF": trainingTimeRBF * 1000,
    "KNN":     trainingTimeKNN * 1000,
    "SGD":     trainingTimeSGD * 1000,
    "RandomForest": trainingTimeRND * 1000
})

ax = train_times_ms.plot(kind="bar", figsize=(10,5))
plt.title("Training Time by Model (ms)")
plt.ylabel("Milliseconds")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Label bars so very small values are still readable
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f} ms",
                (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", xytext=(0,3), textcoords="offset points")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
