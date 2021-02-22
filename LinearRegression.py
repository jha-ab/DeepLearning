#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:17:23 2020

@author: abhishek
"""

import numpy as np
import pylab as py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import interpolate


plt.close('all')
M =  3 #regression model order
k = 1 #Huber M-estimator tuning parameter (Used as variance for RBF kernel)


data_uniform = np.load('TrainData.npy')
x1 = data_uniform[:,0]
t1 = data_uniform[:,1]
test_data = np.load('TestData.npy')  
x2 = test_data[:,0]
t2 = test_data[:,1]

plt.scatter(x1,t1, label='Training data', color='k', marker = '*', s=6.66)
plt.xlim(-4.5,4.5)
plt.ylim(-2,2)
plt.xlabel('Training Data - X')
plt.ylabel('Training Output - t')
plt.title('Training Samples')
plt.legend()
plt.show()
print("Mean is",np.mean(t1))
print("Variance is",np.var(t1, dtype=np.float64))

X = np.array([x1**m for m in range(M+1)]).T
print("X size is ",X.shape)
w = np.linalg.inv(X.T@X)@X.T@t1
#print(w.shape)
X1 = np.array([x2**m for m in range(M+1)]).T
esty = X1@w

plt.scatter(x2,t2, label='Test Data', color='r', marker = 'o', s=0.5)
plt.scatter(x2, esty, label='Estimated Function after polynomial fitting M=3', color='g', marker = '*', s=0.5)
plt.xlim(-4.5,4.5)
plt.ylim(-2,2)
plt.xlabel('Test Data - X')
plt.ylabel('Test Output - t')
plt.title('Test Samples')
plt.legend()
plt.show()



low_mean = -0.227
mid_mean = 0
high_mean = 0.227
#print(x1[1])
mean_array = []
#print(x1)
#print(x1 + 3)
#print(x1)
#print(x1 - 3)
#print("------------------------------------------------------------")
#print(np.exp((-1/2*k**2)*(x1+3)**2))
#print(np.exp((-1/2*k**2)*(x1)**2))
#print(np.exp((-1/2*k**2)*(x1-3)**2))

ones = np.ones(50)
PHI1 = np.exp((-1/2*k**2)*(x1-low_mean)**2)
PHI2 = np.exp((-1/2*k**2)*(x1)**2)
PHI3 = np.exp((-1/2*k**2)*(x1-high_mean)**2)

#PHI = np.concatenate((ones), (PHI1), (PHI2), (PHI3))
#print("Sizes",PHI1.size,PHI2.size,PHI3.size)

PHI = np.vstack((ones,PHI1,PHI2,PHI3))
w_rbf = (np.linalg.inv(PHI@PHI.T)@PHI)@t1
print(w_rbf)

y_rbf = X1@w_rbf

plt.scatter(x2,t2, label='Test Data', color='r', marker = 'o', s=0.5)
plt.scatter(x2, y_rbf, label='Estimated Function after RBF', color='g', marker = '*', s=0.5)
plt.xlim(-4.5,4.5)
plt.ylim(-7.5,7.5)
plt.xlabel('Test Data - X')
plt.ylabel('Test Output - t')
plt.title('Test Samples')
plt.legend()
plt.show()


#PHI = np.vstack((PHI2,PHI3))
#PHI = np.append(PHI, ones.T, axis=0)
#PHI = np.append(PHI, PHI1.T, axis=0)
#PHI = np.append(PHI, PHI2.T, axis=0)
#PHI = np.append(PHI, PHI3.T, axis=0)




#for el in x1:
#    if(el >= -4 and el < -1):
#        mean=np.append('[-3]',axis=0)
#    elif(el >=-1 and el <=1):
#        mean=np.append('[0]',axis=0)
#    elif(el > 1 and el < 4):
#        mean=np.append('[3]',axis=0)
#
#print(mean)
#print(mean.size)

#print(low_mean,mid_mean,high_mean)

#for el in x1:
#    if el >= -4 and el < -1:
#        mean_array = np.append(mean_array, low_mean)
#    if el >= -1 and el < 1:
#        mean_array = np.append(mean_array, mid_mean)
#    if el >= 1 and el < 4:
#        mean_array = np.append(mean_array, high_mean)
#        
    
#PHI = np.exp((-1/2*k**2)*(mean_array-x1)**2)
#np.append(rbfunk,'1')
#
#PHI = np.insert(PHI,0,1,axis=None)
#PHI = np.delete(PHI, 50)
#ivy = PHI.T@PHI
#q = PHI.T@t1
#wt_rbf = q/ivy

#print(wt_rbf)




#print(np.insert(rbfunk,0,1,axis=None).delete(51))


#print(rbfunk.size, rbfunk.shape)
#print(t1.size, t1.shape)
#
#rbfwt = (rbfunk.T@rbfunk)
#rbf_w = rbfunk.T@t1
#print(rbfwt, rbf_w)


#wt_rbf = rbfunk.T@rbfunk@rbfunk.T@t1

#print(wt_rbf)