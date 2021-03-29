# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 14:14:00 2021

@author: flori
"""

from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint





def create_K_folds(K, X, y):
    
    kfold_func = KFold(n_splits=K, shuffle=True)
    
    train_valid_indeces = kfold_func.split(X,y)
    
    return(train_valid_indeces)


def create_list_with_folds(X_train, y_train, n_folds, data_scaler):
    
    train_valid_folds = create_K_folds(5, X_train, y_train)
    l_dicts_fold = []
    
    for train_i, valid_i in train_valid_folds:
        
        # fit on training folds only
        data_scaler.fit(X_train.loc[train_i,: ])
        
        X_train_scaled_fold = data_scaler.transform(X_train.loc[train_i,: ])
        X_valid_scaled_fold = data_scaler.transform(X_train.loc[valid_i,: ])
        y_train_fold = y_train[train_i]
        y_valid_fold = y_train[valid_i]
        
        dict_fold = {
        'X_train_scaled_fold': X_train_scaled_fold,
         'X_valid_scaled_fold': X_valid_scaled_fold,
         'y_train_fold': y_train_fold,
         'y_valid_fold': y_valid_fold
         }
        
        l_dicts_fold.append(dict_fold)
    
    return l_dicts_fold


def apply_CV_model(model, X_train, y_train, model_cv, n_folds, data_scaler):
    
    # create list with folds
    list_with_folds = create_list_with_folds(X_train, y_train, n_folds, data_scaler)
    
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
    loss_per_fold = []
    
    for fold in list_with_folds:
        
        instance_model = model_cv
        
        # train the  model, optimizing with validation
        history_model = instance_model.fit(fold['X_train_scaled_fold'], fold['y_train_fold'], 
                                 validation_data=(fold['X_valid_scaled_fold'],fold['y_valid_fold']),
                  epochs=100, batch_size=2, callbacks=[early_stop_callBack])
        
        loss = model_cv.evaluate(fold['X_valid_scaled_fold'], fold['y_valid_fold'], verbose=0)
        
        print("Fold done.\n Loss: ", loss)
        
        loss_per_fold.append(loss)

    return loss_per_fold
            
# function that retrieves from CV results the CV estimates (average losses)
def get_cv_estimates(cv_results):
    
    # get number of folds
    k = len(cv_results)
    
    loss1 = []
    loss2= []
    
    # gather CV results
    for i in range(k):
        loss1.append(cv_results[i][0])
        loss2.append(cv_results[i][1])
        
    # take means
    return(sum(loss1)/k, sum(loss2)/k)