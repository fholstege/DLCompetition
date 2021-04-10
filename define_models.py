# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:37:39 2021

@author: flori
"""



from tensorflow.keras.losses import MeanSquaredLogarithmicError
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Input, Dropout, maximum
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from sklearn import preprocessing
from keras.models import load_model
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.constraints import max_norm
import keras
from keras.layers import LeakyReLU



def get_base_model(input_dim, base_n_nodes, multiplier_n_nodes):
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


    Returns
    -------
    model: a model object.

   """
   
    n_second_layer = int(round(base_n_nodes* multiplier_n_nodes))

    model = Sequential()
    model.add(Dense(base_n_nodes, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_second_layer, activation='relu'))
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer='Adam', loss=MeanSquaredLogarithmicError(), metrics=['mean_absolute_error'])
    return model


def get_base_model_with_dropout(input_dim, base_n_nodes, multiplier_n_nodes, prob_dropout):
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
    
    prob_dropout: float, [0,1]
        probability of dropping out in layer


    Returns
    -------
    model: a model object.

   """
   
    n_second_layer = base_n_nodes* multiplier_n_nodes

    model = Sequential()
    
    # define first layer
    model.add(Dense(base_n_nodes,input_dim=input_dim,kernel_initializer = 'normal' ,activation='relu'))
    
    # drop out after first layer
    model.add(Dropout(prob_dropout))
    
    
    # add another layer and dropout
    model.add(Dense(n_second_layer, activation='relu'))
    model.add(Dropout(prob_dropout))
    
    # put out final prediction
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer='Adam', loss=MeanSquaredLogarithmicError(), metrics=['mean_absolute_error'])
    return model



def get_base_model_with_maxout(input_dim, base_n_nodes, multiplier_n_nodes, prob_dropout, c, lr):
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
    
    prob_dropout: float, [0,1]
        probability of dropping out in layer


    Returns
    -------
    model: a model object.

   """
   
    # define optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
   
    n_second_layer = int(round(base_n_nodes* multiplier_n_nodes))

    model = Sequential()
    
    # define first layer
    
    model.add(Dense(base_n_nodes, input_dim=input_dim, kernel_initializer = 'normal', 
                    activation='relu', kernel_constraint=max_norm(c)))
    

    # drop out after first layer
    model.add(Dropout(prob_dropout))
    
    # use first maxout layer
    model.add(tfa.layers.Maxout(int(base_n_nodes*prob_dropout), axis=-1 ))
    
    # add another layer and dropout
    model.add(Dense(int(multiplier_n_nodes*base_n_nodes), activation='relu', kernel_constraint=max_norm(c)))

    # use second dropout
    model.add(Dropout(prob_dropout))
     
    # use second maxout layer
    model.add(tfa.layers.Maxout(int(multiplier_n_nodes**2 * base_n_nodes)))
    
    # put out final prediction
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer=opt, loss=MeanSquaredLogarithmicError(), metrics=['mean_absolute_error'])
    return model


def get_base_model_with_maxnorm(input_dim, base_n_nodes, multiplier_n_nodes, prob_dropout, c, lr, alpha):
    """
    

    Parameters
    ----------
    input_dim : int
        How many independent variables.
    base_n_nodes : int
        How many nodes in input layer.
    multiplier_n_nodes : float
        size of layer i is: layer i -1 * multiplier_n_nodes.
        Example; if input is 36, then first hidden layer is 36 * multiplier_n_nodes large
    prob_dropout : float
        Float between [0,1], probability of dropout
    c : int
        max norm restriction on weights.
    lr : float
        learning rate for adam optimizer.
    alpha : float
        parameter for leakyy relu.

    Returns
    -------
    model; instance of model created.

    """
   
   
    # define optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
   
    # define the number of nodes in the second layer
    n_second_layer = int(round(base_n_nodes* multiplier_n_nodes))

    # start model, add layers here
    model = Sequential()
    
    # define first layer - normal distribution for weight initialization, constraint by max norm
    model.add(Dense(base_n_nodes,input_dim=input_dim, kernel_initializer = 'normal', 
                     kernel_constraint=max_norm(c)))
    
    # add leakyrelu activation
    model.add(LeakyReLU(alpha=alpha))
    
    # drop out after first layer
    model.add(Dropout(prob_dropout))
    
    # add hidden layer and dropout
    model.add(Dense(n_second_layer,  kernel_constraint=max_norm(c)))
    
    # add leaky relu activation
    model.add(LeakyReLU(alpha=alpha))
    
    # dropout after hidden layer
    model.add(Dropout(prob_dropout))
    
    # put out final prediction
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer=opt, loss=MeanSquaredLogarithmicError(), metrics=['mean_absolute_error'])
    return model



    
    