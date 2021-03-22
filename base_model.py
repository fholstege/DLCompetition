# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:33:08 2021

@author: flori
"""

import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.losses import MeanSquaredLogarithmicError

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


# this is the training data - we need to split this in train, validation.
# We also should get a test sample from this dataset, to see how we perform out of sample berfore submission
df_train_data = pd.read_csv("https://www.dropbox.com/s/bawlkeolef1bse2/train_dat.csv?dl=1", 
                        sep= ",")

# this is the test data for the competition - we save this here for later
df_test_data = pd.read_csv("https://www.dropbox.com/s/rbjatpuk5x7dios/test_dat.csv?dl=1", 
                         sep= ",")

# define what percentage for validation, what percentage for test - we use this for training
validation_perc = 0.2
test_perc = 0.1

# define random state
state = 123

# define the independent and dependent variable
df_X = df_train_data.iloc[:,df_train_data.columns != 'y_train']
df_Y = df_train_data['y_train']

# turn to numpy arrays
X = np.array(df_X.values)
y = np.array(df_Y.values)

# get dataframes for training and test, and then for validation
X_train_valid, X_test, y_train_valid, y_test  = train_test_split(X, y, test_size= test_perc, random_state = state)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = validation_perc, random_state = state )


def define_model(input_dim, output_dim, base_n_nodes, multiplier_n_nodes, prob_dropout):
    """

    Parameters
    ----------
    input_dim : TYPE
        DESCRIPTION.
    output_dim : TYPE
        DESCRIPTION.
    base_n_nodes : TYPE
        DESCRIPTION.
    multiplier_n_nodes : TYPE
        DESCRIPTION.
    prob_dropout : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

   """
    

    inputs = Input(shape=(input_dim,))
    l = BatchNormalization()(inputs)
    l = Dropout(prob_dropout)(l)
    n = base_n_nodes
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(prob_dropout)(l)
    n = int(n * multiplier_n_nodes)
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(prob_dropout)(l)
    n = int(n * multiplier_n_nodes)
    l = Dense(n, activation='relu')(l)
    outputs = Dense(output_dim, activation='softmax')(l)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss=MeanSquaredLogarithmicError)
    return model



model_NN = get_model(len(list_features) - 1, 6, 13, 0.75, 0.5)
model_NN.summary()