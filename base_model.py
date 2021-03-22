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
from sklearn import preprocessing

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

        
l_continuous_features = ['OverallQual', 'OverallCond', 
       'TotalBsmtSF', 'X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
       'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal' ]

# select continuous features, and scale these
df_X_selectedFeatures = df_X[l_continuous_features]
df_X_scaled = preprocessing.StandardScaler().fit_transform(df_X_selectedFeatures)

# define dataframe with y variable
df_Y = df_train_data['y_train']


# turn to numpy arrays
X = df_X_scaled
y = np.array(df_Y.values)

# get dataframes for training and test, and then for validation
X_train_valid, X_test, y_train_valid, y_test  = train_test_split(X, y, test_size= test_perc, random_state = state)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = validation_perc, random_state = state )



#####################
# In this part of the code, we try out the most basic model - only continuous features, 2 layers
#
#####################



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


# base model - can tweak hyper parameters
base_model = get_base_model(input_dim=24, base_n_nodes=24, multiplier_n_nodes = 0.5, prob_dropout=0.2)
base_model.summary()

# save the weights of the model here
checkpoints = ModelCheckpoint(
         'base_model_weights.hdf5', 
          save_best_only=True, 
          save_weights_only=True,
          verbose=1)
# implement early stop - prevents it from continuing after no improvement in validation
early_stop = EarlyStopping(patience=20) 

# train the model, optimizing with validation
history = base_model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          epochs=100, batch_size=1, callbacks=[early_stop ,checkpoints])


# load the best weights from the model, then check how performs in all sets
# test set is here our own defined one

base_model.load_weights('base_model_weights.hdf5')

print('Training Loss : {}\nValidation Loss : {} \n Test Loss: {}'.format(base_model.evaluate(X_train, y_train), base_model.evaluate(X_valid, y_valid), base_model.evaluate(X_test, y_test)))

predictions_base = base_model.predict(X_test)
plt.hist(predictions_base)
