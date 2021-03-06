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
from sklearn.model_selection import train_test_split

# import custom models defined in define_models
from define_models import get_base_model, get_base_model_with_dropout, get_base_model_with_maxout, get_base_model_with_maxnorm
from help_functions import apply_CV_model, get_cv_estimates, create_list_with_folds, create_K_folds

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
y_train_logged = np.log(y_train)

print("N of variables: ", len(X_train.columns))

# remove two outliers
train_df_noOutliers = train_df[(train_df['LotArea'] < 200000) & (train_df['TotalBsmtSF'] < 6000)]

# define training vars
X_train_noOutliers = train_df_noOutliers.drop("y_train", axis = 1)
y_train_noOutliers = train_df_noOutliers["y_train"]


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
results_cv_base = apply_CV_model(X_train_noOutliers, y_train_noOutliers, base_model, K_folds, data_scaler)
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
base_model_maxnorm.summary()

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


###################
# The base_model_maxnorm seems to perform best - here we try different values for the p for it
# 
###################

# probabilities to try out
probabilities_dropout = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65, 0.7,0.75,0.8]


def try_different_p(lProbabilities):
    
    # get result of cv per probability
    result_per_prob = []
    
    # loop through each probability
    for experimental_prob_dropout in lProbabilities:
        
        print("Trying out a p of: ", experimental_prob_dropout)
        
        # define the model
        base_model_maxnorm_probExp = get_base_model_with_maxnorm(input_dim=n_inputs, base_n_nodes=n_inputs/(1-experimental_prob_dropout), multiplier_n_nodes = 0.5, prob_dropout = experimental_prob_dropout, c = 4.0, lr = 0.01)
        
        # get cv results for the p
        results_cv_base_dropout_maxnorm_probExp  = apply_CV_model(X_train_noOutliers, y_train_noOutliers, base_model_maxnorm_probExp, K_folds, data_scaler)
        
        # get msle for the cv
        msle_cv_base_dropout_maxnorm_probExp = [item[0] for item in results_cv_base_dropout_maxnorm_probExp]
        
        # get mean result and save
        result_per_prob.append(np.mean(msle_cv_base_dropout_maxnorm_probExp))
        
        print("Result: ", np.mean(msle_cv_base_dropout_maxnorm_probExp))
    
    return result_per_prob

result_per_prob_leakyRelu = try_different_p(probabilities_dropout)

plt.plot(probabilities_dropout,result_per_prob_leakyRelu )




#################
# Here we generate predictions
#################


# first, check that the train and test df are similar
X_train_noOutliers.columns == test_df.columns

 # create a single training and validation split
X_train_final, X_valid_final, y_train_final, y_valid_final = train_test_split(X_train_noOutliers, y_train_noOutliers, test_size=validation_perc, random_state=123)


# scale the independent variables for training and validation
data_scaler.fit(X_train_final)
X_train_final_scaled = data_scaler.transform(X_train_final)
X_valid_final_scaled = data_scaler.transform(X_valid_final)

# scale the independent variables for the test set
X_test_final_scaled = data_scaler.transform(test_df)

# best dropout based on cv
best_dropout = 0.5
final_model = get_base_model_with_maxnorm(input_dim=n_inputs, base_n_nodes=n_inputs/(1-best_dropout), multiplier_n_nodes = 0.5, prob_dropout = best_dropout, c = 4.0, lr = 0.01)
     
final_model.summary()  

# early stopping to save best model 
early_stop_callBack =  EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1,
                            mode='auto',
                            baseline=None,
                            restore_best_weights=True
                            )



# train the  model, optimizing with validation, exactly like done in the CV
history_model = final_model.fit(X_train_final_scaled, y_train_final, validation_data=(X_valid_final_scaled,y_valid_final),
                  epochs=100, batch_size=2, callbacks=[early_stop_callBack])

# get loss of best model on validation set
loss_validation = final_model.evaluate(X_valid_final_scaled, y_valid_final, verbose=0)

   
# get predictions for the test set
predictions_test_set = final_model.predict(X_test_final_scaled)
predictions_valid_set = final_model.predict(X_valid_final_scaled)
predictions_training_set = final_model.predict(X_train_final_scaled)

plt.hist(predictions_test_set)
plt.hist(predictions_valid_set)
plt.hist(predictions_training_set)

type(predictions_test_set)
# write predictions to csv
pd.DataFrame(predictions_test_set).to_csv("FlorisHolstege_MarkusMueller.csv",header=False, index = False )

test_df = pd.read_csv("FlorisHolstege_MarkusMueller.csv")

plt.hist(test_df)