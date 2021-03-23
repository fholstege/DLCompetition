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
from keras.callbacks import EarlyStopping, ModelCheckpoint
import smogn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error

# import custom models defined in define_models
from define_models import get_base_model, get_base_model_with_dropout


filepath_models = 'files_models'

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
        
l_continuous_features = ['OverallQual', 'OverallCond', 
       'TotalBsmtSF', 'X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
       'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal' ]

# select continuous features, and scale these
df_X_selectedFeatures = df_train_data[l_continuous_features]

df_X_scaled = preprocessing.StandardScaler().fit_transform(df_X_selectedFeatures)

# define dataframe with y variable
df_Y = df_train_data['y_train']


# turn to numpy arrays
X = df_X_scaled
y = np.array(df_Y.values)

# get dataframes for training and test, and then for validation
X_train_valid, X_test, y_train_valid, y_test  = train_test_split(X, y, test_size= test_perc, random_state = state)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = validation_perc, random_state = state )

# apply smogn to oversample
df_train_smogn = pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)], axis = 1)
df_train_smogn.columns = l_continuous_features + ['y_train']
df_train_synthetic = smogn.smoter(data =df_train_smogn, y='y_train', samp_method='balance' )

# get X train and y train from oversampling class 
X_train_synthetic = np.array(df_train_synthetic[l_continuous_features])
y_train_synthetic = np.array(df_train_synthetic['y_train'])

#####################
# In this part of the code, we try out the most basic model - only continuous features, 2 layers
#
#####################


# base model 
base_model = get_base_model(input_dim=24, base_n_nodes=24, multiplier_n_nodes = 0.5)
base_model.summary()

# save the weights of the best base model here
checkpoints_base = ModelCheckpoint(
          filepath_models + 'base_model_weights.hdf5', 
          save_best_only=True, 
          save_weights_only=True,
          verbose=1)
# implement early stop - prevents it from continuing after no improvement in validation
early_stop = EarlyStopping(patience=20) 

# train the base model, optimizing with validation
history_base = base_model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          epochs=100, batch_size=1, callbacks=[early_stop ,checkpoints_base])


# load the best weights from the model, then check how performs in all sets
# test set is here our own defined one

base_model.load_weights(filepath_models + 'base_model_weights.hdf5')

print('Results Base model - \n Training Loss : {}\nValidation Loss : {} \n Test Loss: {}'.format(base_model.evaluate(X_train, y_train), base_model.evaluate(X_valid, y_valid), base_model.evaluate(X_test, y_test)))


#####################
# In this part of the code, we extend the most basic model - add dropout
#
#####################

# base model extended 
base_model_extended = get_base_model_with_dropout(input_dim=24, base_n_nodes=36, multiplier_n_nodes = 0.5, prob_dropout = 0.3)
base_model_extended.summary()

# save the weights of the model here
checkpoints_extended = ModelCheckpoint(
          filepath_models + 'base_model_extended_weights.hdf5', 
          save_best_only=True, 
          save_weights_only=True,
          verbose=1)

# train the model, optimizing with validation
history_extended = base_model_extended.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          epochs=100, batch_size=1, callbacks=[early_stop ,checkpoints_extended])


# load the best weights from the model, then check how performs in all sets
# test set is here our own defined one
base_model_extended.load_weights(filepath_models + 'base_model_extended_weights.hdf5')

print('Results extended model - \n Training Loss : {}\nValidation Loss : {} \n Test Loss: {}'.format(base_model_extended.evaluate(X_train, y_train), base_model_extended.evaluate(X_valid, y_valid), base_model_extended.evaluate(X_test, y_test)))

#####################
# In this part of the code, we add synthetic data to oversample from rare classes from
#
#####################

# save the weights of the model here
checkpoints_synthetic = ModelCheckpoint(
          filepath_models + 'base_model_synthetic_weights.hdf5', 
          save_best_only=True, 
          save_weights_only=True,
          verbose=1)

# base model with synthetic
base_model_synthetic = get_base_model(input_dim=24, base_n_nodes=24, multiplier_n_nodes = 0.5)
base_model_synthetic.summary()


# train the model, optimizing with validation
history_synthetic = base_model_synthetic.fit(X_train_synthetic, y_train_synthetic, validation_data=(X_valid, y_valid),
          epochs=100, batch_size=1, callbacks=[early_stop ,checkpoints_synthetic])


# load the best weights from the model, then check how performs in all sets
# test set is here our own defined one
base_model_synthetic.load_weights(filepath_models + 'base_model_synthetic_weights.hdf5')

print('Base model  with synthetic data - \n Training Loss : {}\nValidation Loss : {} \n Test Loss: {}'.format(base_model_extended.evaluate(X_train, y_train), base_model_extended.evaluate(X_valid, y_valid), base_model_extended.evaluate(X_test, y_test)))


#####################
# In this part of the code, we log the dependent - subsequently, we revert it back, and test the prediction accuracy
#
#####################

# save the weights of the model here
checkpoints_logged = ModelCheckpoint(
          filepath_models + 'base_model_logged_weights.hdf5', 
          save_best_only=True, 
          save_weights_only=True,
          verbose=1)

# base model with logged data
base_model_logged = get_base_model(input_dim=24, base_n_nodes=24, multiplier_n_nodes = 0.5)
base_model_logged.summary()


# train the model, optimizing with validation
history_logged = base_model_logged.fit(X_train, np.log(1 + y_train), validation_data=(X_valid, np.log(1 + y_valid)),
          epochs=100, batch_size=1, callbacks=[early_stop ,checkpoints_logged])


# load the best weights from the model, then check how performs in all sets
# test set is here our own defined one
base_model_logged.load_weights(filepath_models + 'base_model_logged_weights.hdf5')

# get logged predictions, backtransform
logged_predictions_train = base_model_logged.predict(X_train)
logged_predictions_valid = base_model_logged.predict(X_valid)
logged_predictions_test = base_model_logged.predict(X_test)

backtransformed_predictions_train = np.exp(logged_predictions_train) - 1
backtransformed_predictions_valid = np.exp(logged_predictions_valid) - 1
backtransformed_predictions_test = np.exp(logged_predictions_test) - 1

msle_train = mean_squared_log_error(y_train, backtransformed_predictions_train)
msle_valid = mean_squared_log_error(y_valid, backtransformed_predictions_valid)
msle_test = mean_squared_log_error(y_test, backtransformed_predictions_test)

print('Base model  with logged data - \n Training Loss : {}\nValidation Loss : {} \n Test Loss: {}'.format(msle_train, msle_valid, msle_test))