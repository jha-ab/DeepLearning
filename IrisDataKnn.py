#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 05:03:52 2020

@author: abhishek
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt


def knn(neighbors, x_train, y_train, x_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)    
    return y_pred

def result(y_test, y_pred):
    print("Accuracy Score", metrics.accuracy_score(y_test, y_pred)) 
    print(classification_report(y_test, y_pred))
    c = confusion_matrix(y_test, y_pred) # Plotting the confusion matrix
    print(c)



if __name__ == "__main__":

    iris = datasets.load_iris()
    
    # Min Max Normalization
    scaler = MinMaxScaler()
    scaler.fit(iris.data)
    X_scaled = scaler.transform(iris.data)
    
    # Z-Score Normalization
    X_z = stats.zscore(iris.data)
    
    fig, axes = plt.subplots(1,3)
    axes[0].scatter(iris.data[:,0], iris.data[:,1], color="k")
    axes[0].set_title("Original data")
    axes[1].scatter(X_scaled[:,0], X_scaled[:,1], c="c")
    axes[1].set_title("Min-Max")
    axes[2].scatter(X_z[:,0], X_z[:,1], c="g")
    axes[2].set_title("Z-Score")
    plt.show()
        
    
    
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, iris.target, test_size=0.7)
    #print(x_train.shape,x_test.shape) # (105, 4) (45, 4)
    x_t, x_tst, y_t, y_tst = train_test_split(x_train, y_train, test_size = 0.2)
    y_p = knn(7, x_train, y_train, x_test)
#    print(y_p)
    result(y_test, y_p)
                    