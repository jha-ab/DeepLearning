#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:09:41 2020

@author: abhishek
"""

import math
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, MaxPooling2D, BatchNormalization, SimpleRNN, LeakyReLU


def generatedata(order):

    train = scipy.io.loadmat('train_tau_30.mat')
    test = scipy.io.loadmat('test_tau_30.mat')
    traindata = np.array(train['X1'])
    testdata = np.array(test['Y1'])
    (X_train, y_train, X_test, y_test) = ([], [], [], [])

    e1 = 0.95 * np.random.normal(0, 0.5, 1000)
    e2 = 0.05 * np.random.normal(1, 0.5, 1000)
    e1 = e1.reshape((1000, 1))
    e2 = e2.reshape((1000, 1))

    e = np.add(e1, e2)

               # print(e.shape)

    for i in range(traindata.shape[0] - order):

        temp = traindata[i:i + order:1]  # , e[i:i+order:1])
        temp2 = traindata[i + order]  # , e[i+order])
        X_train.append(temp)

                              # y_train.append(traindata[i+order])

        y_train.append(temp2)
        X_test.append(testdata[i:i + order:1])
        y_test.append(testdata[i + order])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    (X_train, X_val, y_train, y_val) = train_test_split(X_train,
            y_train, test_size=0.20, random_state=7)

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return (
        X_train,
        X_test,
        X_val,
        y_val,
        y_train,
        y_test,
        )



def modelinit(lr, batch_size):

    model = keras.Sequential()
    model.add(Conv1D(filters=2, kernel_size=order, activation=None,
              strides=1, input_shape=(order, 1)))
#    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten(input_shape=(order, 1)))
    model.add(Dense(1, activation=None))
               # Set learning rate and optimizer

    opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9,
                                   beta_2=0.999, epsilon=1e-08,
                                   amsgrad=False)

    def max_error_entropy(y_actual, y_pred):
        error = y_actual - y_pred
        error_T = K.repeat(error, K.shape(error)[0])
        mod = tf.cast(tf.shape(y_actual)[0], 'float32')
        pi = tf.cast(np.pi, 'float32')
        return -K.sum(K.exp(-K.square(error - error_T) / 2
                      * kernel_size ** 2) / (K.sqrt(2 * pi)
                      * kernel_size * mod ** 2))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,   #tf.keras.losses.MeanSquaredError()
                  metrics=['accuracy'])
    return model


def model_fit(
    model,
    X_train,
    X_val,
    y_train,
    y_val,
    batch_size,
    epochs,
    ):

    (scores, histories) = (list(), list())

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
        )

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=1,
        )

               # evaluate model

    (_, acc) = model.evaluate(X_test, y_test, verbose=0)

    print('Val acc = %.3f' % (acc * 100.0))

               # stores scores

    scores.append(acc)

    histories.append(history)

    model.summary()

    return (scores, histories)


def summarize_diagnostics(histories, lr):

               # for i in range(len(histories)):

               # plot loss

    loss = np.asarray(histories[0].history['loss']) * -1
    val_loss = np.asarray(histories[0].history['val_loss']) * -1
    plt.figure()
    plt.title('MSE Loss')
    plt.plot(loss, color='blue', label='train')
    plt.plot(val_loss, color='orange', label='test')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'val'], loc='best')
               # plt.savefig('srnn_lr_e{}.png'.format(lr))
    plt.show()


def model_evaluate(
    model,
    X_test,
    y_test,
    lr,
    ):

               # Get predictions

    predictions = model.predict(X_test)
    predictions = np.array(predictions)
              # print(predictions.shape)
    plt.figure()
    plt.title('predicted vs actual lr:{}'.format(lr))
    plt.plot(predictions, color='orange', label='predicted')
    plt.plot(y_test, color='blue', label='actual')
    plt.ylabel('f(n)')
    plt.xlabel('time')
    plt.legend(['predicted', 'actual'], loc='best')


if __name__ == "__main__":
    #!/usr/bin/python
# -*- coding: utf-8 -*-
    order = 10
    (
        X_train,
        X_test,
        X_val,
        y_val,
        y_train,
        y_test,
        ) = generatedata(order)
    lr = 0.067
    sigma = 0.7
    kernel_size = 2 ** 0.5 * sigma
    batch_size = 32
    epochs = 1000
    
    model = modelinit(lr, batch_size)
    (scores, histories) = model_fit(
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        batch_size,
        epochs,
        )
    
    # Evaluate model
    
    model_evaluate(model, X_test, y_test, lr)
    
    # Plots
    
    summarize_diagnostics(histories, lr)

