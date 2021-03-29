# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:33:08 2021

@author: flori
"""

from pathlib import Path
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
from pathlib import Path
import os 

# import custom models defined in define_models
from define_models import get_base_model, get_base_model_with_dropout, get_base_model_with_maxout, get_base_model_with_maxnorm
from help_functions import apply_CV_model, get_cv_estimates

# define path for models
filepath_models = 'files_models/'

# define data path
data_path = Path(os.getcwd()).parent/"DLCompetition/data"


# load train and test set
train_df = pd.read_hdf(data_path/'train_df.hdf5')
test_df = pd.read_hdf(data_path/'test_df.hdf5')

# define what percentage for validation
validation_perc = 0.2

# define random state
state = 123

# define training vars
X_train = train_df.drop("y_train", axis = 1)
y_train = train_df["y_train"]

# define the data scaler - mean 0, var of 1
data_scaler = preprocessing.StandardScaler()

# n inputs for the model 
n_inputs = X_train.shape[1]

# probability of droput
prob_drop = 0.5
K_folds = 5

#####################
# Base model: we try out the most basic model 2 layers
#
#####################

# base model with two layers 
base_model = get_base_model(input_dim=n_inputs, base_n_nodes=n_inputs, multiplier_n_nodes = 0.5)


# apply cv here 
results_cv_base = apply_CV_model(X_train, y_train, base_model, K_folds, data_scaler)

msle_cv_base= [item[0] for item in results_cv_base]


#####################
# In this part of the code, we extend the most basic model - add dropout
#
#####################

# base model extended with dropout
base_model_extended = get_base_model_with_dropout(input_dim=n_inputs, base_n_nodes=n_inputs/(1-prob_drop), multiplier_n_nodes = 0.5, prob_dropout = 0.5)

results_cv_base_dropout  = apply_CV_model(X_train, y_train, base_model_extended, K_folds, data_scaler)

msle_cv_base_dropout= [item[0] for item in results_cv_base_dropout]

#####################
# In this part of the code, we use the relu activation combined with dropout and maxnorm constraint (c)
#
#####################

# base model with max norm
base_model_maxnorm = get_base_model_with_maxnorm(input_dim=n_inputs, base_n_nodes=n_inputs/(1-prob_drop), multiplier_n_nodes = 0.5, prob_dropout = prob_drop, c = 4.0, lr = 0.01)

results_cv_base_dropout_maxnorm  = apply_CV_model(X_train, y_train, base_model_maxnorm, K_folds, data_scaler)

msle_cv_base_dropout_maxnorm = [item[0] for item in results_cv_base_dropout_maxnorm]

#####################
# In this part of the code, we use the maxout activation combined with dropout (also maxnorm and relu)
#
####################

# base model with maxout
base_model_maxout = get_base_model_with_maxout(input_dim=n_inputs, base_n_nodes= n_inputs/(1-prob_drop), multiplier_n_nodes = 0.5, prob_dropout = 0.5, c = 4.0, lr = 0.01)

results_cv_base_dropout_maxout  = apply_CV_model(X_train, y_train, base_model_maxout, K_folds, data_scaler)

msle_cv_base_dropout_maxout = [item[0] for item in results_cv_base_dropout_maxout]


# compare all results; mean msle across models
np.mean(msle_cv_base)
np.mean(msle_cv_base_dropout)
np.mean(msle_cv_base_dropout_maxnorm)
np.mean(msle_cv_base_dropout_maxout)
