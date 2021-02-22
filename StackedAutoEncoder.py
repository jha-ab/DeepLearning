#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 00:41:58 2020

@author: abhishek
"""

import keras
from   keras import layers as l
from   keras.models import Sequential
from   keras.layers.core import Dense, Activation, Flatten, Dropout
from   keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np  
import matplotlib.pyplot as plt
from   keras.datasets import fashion_mnist
from   tensorflow.keras.optimizers import Adam, Nadam
from   tensorflow.keras.regularizers import l2
import itertools
from   sklearn.metrics import confusion_matrix
from   sklearn.model_selection import train_test_split
from   sklearn.utils import shuffle
#import scikitplot as skplt

#Hyperparameters
X          = 5        # Bottleneck layer units
batch_size = 200
lr_act     = 'tanh'     #Layer activations
op_act     = 'sigmoid'  #Output activation

# Stacked Encoder Hyperparameters
epochs1    = 20
lr1        = 0.9

# Encoder only Hyperparameters
epochs2    = 100
lr2        = 0.0001

# Data pre-processing
(x_train , y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
x = np.concatenate((x_train,x_test),axis=0)
y = np.concatenate((y_train,y_test),axis=0)

# Shuffle the dataset
x,y = shuffle(x, y, random_state=2)
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

y_train   = keras.utils.to_categorical(y_train, num_classes=None, dtype="int")
y_test_cm = y_test
y_test    = keras.utils.to_categorical(y_test, num_classes=None, dtype="int")

# ----------------------* Flatten the dataset *---------------------------#
temp = np.zeros((x_train.shape[0],784),dtype=float)
for i in range(x_train.shape[0]):
    temp[i,:] = x_train[i,:].flatten(order='C')
x_train = temp

temp = np.zeros((x_test.shape[0],784),dtype=float)
for i in range(x_test.shape[0]):
    temp[i,:] = x_test[i,:].flatten(order='C')
x_test = temp    

# Normalizing the input
x_train = x_train/255
    
# ----------------------* Defining SAE Architecture *---------------------#
SAE = Sequential([   
Dense(units=500, input_shape=(784,), use_bias=False),
l.BatchNormalization(),
Activation(lr_act),
Dense(units = 200, use_bias=False),
l.BatchNormalization(),
Activation(lr_act),
Dense(units=X, use_bias=False),
l.BatchNormalization(),
Activation(lr_act),
Dense(units = 200, use_bias=False),
l.BatchNormalization(),
Activation(lr_act),
Dense(units = 500, use_bias=False),
l.BatchNormalization(),
Activation(lr_act),
Dense(units = 784, use_bias=False),
Activation(op_act),
])

filepath = 'weights_checkpoint'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = filepath,
                                                             save_weights_only=True,
                                                             monitor='accuracy',
                                                             mode='max',
                                                             save_best_only=False)

def training_step1(epochs,lr):
    SAE.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error',
                metrics=['accuracy'])
    SAE.summary()
    x = SAE.fit(x_train,
                x_train,
                validation_split=0.1,
                epochs=epochs,
                shuffle=True,
                batch_size=batch_size,
                )
    return x
        
history1 =  training_step1(epochs1, lr1)
 

# Saving the encoder weights
k1 = np.array(SAE.layers[0].get_weights())
k2 = np.array(SAE.layers[1].get_weights())
k3 = np.array(SAE.layers[3].get_weights())
k4 = np.array(SAE.layers[4].get_weights())
k5 = np.array(SAE.layers[6].get_weights())

# np.save('k1',k1)
# np.save('k2',k2)
# np.save('k3',k3)
# np.save('k4',k4)
# np.save('k5',k5)


#------------------------* Check SAE output *-----------------------------#
def img_pred():
    from random import randrange
    n = 5
    k = np.zeros(n,dtype=int)
    for i in range(n):
        k[i] = randrange(10)*randrange(10)*randrange(10)*randrange(10) 
    ip = x_test[k,:]
    op = SAE.predict(ip)
    
    temp = np.zeros((n,28,28),dtype=float)
    for p in range(n):
        for i in range(28):
            for j in range(28):
                temp[p,i,j] = ip[p,i*28 + j]
    ip = temp
    temp = np.zeros((n,28,28),dtype=float)
    for p in range(n):
        for i in range(28):
            for j in range(28):
                temp[p,i,j] = op[p,i*28 + j]
    op = temp
    
    for i in range(n):
        plt.subplot(2, n, 1+i)
        plt.imshow(ip[i,:,:])
        plt.subplot(2, n, (n+1+i)) 
        plt.imshow(op[i,:,:])


# -------------------------* No Decoder*** ---------------------------------#

SE = Sequential([  
# ------------------------* Encoder Start *-------------------------------#    
Dense(units=500, input_shape=(784,), use_bias=False),
l.BatchNormalization(),
Activation(lr_act),
Dense(units = 200, use_bias=False),
l.BatchNormalization(),
Activation(lr_act),
Dense(units=X, use_bias=False),
# -----------------------* Encoder end *----------------------------------#
# -------------------------* MLP START *----------------------------------#
Dense(units=100, use_bias=(False)),
Activation(lr_act),
Dropout(0.1),
Dense(units=50, use_bias=(False)),
Activation(lr_act),
Dropout(0.2),
Dense(units=10, use_bias=(False)),
Activation('softmax'),
])


# Loading and assigning trained encoder weights
SE.layers[0].set_weights(k1)
SE.layers[1].set_weights(k2)
SE.layers[3].set_weights(k3)
SE.layers[4].set_weights(k4)
SE.layers[6].set_weights(k5)

# Making Encoder weights non trainable
SE.layers[0].trainable = False 
SE.layers[1].trainable = False 
SE.layers[3].trainable = False 
SE.layers[4].trainable = False 
SE.layers[6].trainable = False 

def training_step2(epochs,lr):
    SE.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy',
               metrics=['accuracy'])
    SE.summary()
    x = SE.fit(x_train,
               y_train,
               validation_split=0.1,
               epochs=epochs,
               shuffle=True,
               batch_size=batch_size,
               )
    return x
        
history2 =  training_step2(epochs2, lr2)
 
val_loss, val_acc = SE.evaluate(x_test, y_test)
print('Loss: ' + str(round(val_loss,5)) + '  |  ' +  'Accuracy: ' + str(round(val_acc,5)))

pred = SE.predict_classes(x=x_test)
cm   = confusion_matrix(y_true=y_test_cm,y_pred=pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix,without normalization')
        
    print(cm)
    
    tresh= cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i,j] > tresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cm.png', dpi = 300)
    

cm_plot_labels = ['Class 1','Class 2','Class 3','Class 4','Class 5',
                  'Class 6','Class 7','Class 8','Class 9','Class 10']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels,title='Confusion Matrix')