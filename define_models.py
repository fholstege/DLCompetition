# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:37:39 2021

@author: flori
"""



from tensorflow.keras.losses import MeanSquaredLogarithmicError
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Input, Dropout
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from sklearn import preprocessing
from keras.models import load_model


def get_base_model(input_dim, base_n_nodes, multiplier_n_nodes, prob_dropout):
    """

    Parameters
    ----------
    input_dim : integer
        The number of independent variables.

    base_n_nodes : integer
        The number of nodes for the first layer
    multiplier_n_nodes : float, [0,1]
        with each layer that we add, 
        we decrease the nodes to the number 
        of nodes in previous layer multiplied with this float, .
    prob_dropout : float, [0,1]
        The probability of dropout.

    Returns
    -------
    model: a model object.

   """
   
    n_second_layer = base_n_nodes* multiplier_n_nodes

    model = Sequential()
    model.add(Dense(base_n_nodes, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_second_layer, activation='relu'))
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer='Adam', loss=MeanSquaredLogarithmicError(), metrics=['mean_absolute_error'])
    return model
