# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:25:25 2021

@author: flori
"""

# strange behaviour: for first run, initial loss is very high like 33, subsequent runs start as much lower loss?



# # define validation set
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = validation_perc, random_state = state)

# # scale training data, use same mean for train and test set
# data_scaler = preprocessing.StandardScaler()
# X_train_sc = data_scaler.fit_transform(X_train)
# X_valid_sc = data_scaler.fit_transform(X_valid)

# # turn into numpy array
# y_train, y_valid = np.array(y_train), np.array(y_valid)

# # using CV would mean to first determine the test fold, then do scaling based on rest of folds

# # apply smogn to oversample
# df_train_smogn = pd.concat([pd.DataFrame(X_train_sc),pd.DataFrame(y_train)], axis = 1)
# df_train_smogn.columns = list(X_train.columns) + ['y_train']
# df_train_synthetic = smogn.smoter(data =df_train_smogn, y='y_train', samp_method='extreme' )

# #  get X train and y train from oversampling class 
# X_train_synthetic = np.array(df_train_synthetic[list(X_train.columns) ])
# y_train_synthetic = np.array(df_train_synthetic['y_train'])





# # # get X train and y train from oversampling class 
# X_train_synthetic = np.array(df_train_synthetic[list(X_train.columns) ])
# y_train_synthetic = np.array(df_train_synthetic['y_train'])




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
base_model_synthetic = get_base_model_with_dropout(input_dim=X_train_sc.shape[1], base_n_nodes=300, multiplier_n_nodes = 0.5, prob_dropout = 0.3)
base_model_synthetic.summary()


# train the model, optimizing with validation
history_synthetic = base_model_synthetic.fit(X_train_synthetic, y_train_synthetic, validation_data=(X_valid_sc, y_valid),
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
