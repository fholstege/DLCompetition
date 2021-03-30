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
import tensorflow_probability as tfp
import keras


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


def get_base_model_with_maxnorm(input_dim, base_n_nodes, multiplier_n_nodes, prob_dropout, c, lr):
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
    model.add(Dense(base_n_nodes,input_dim=input_dim, kernel_initializer = 'normal', 
                    activation='relu', kernel_constraint=max_norm(c)))
    
    # drop out after first layer
    model.add(Dropout(prob_dropout))
    
    # add another layer and dropout
    model.add(Dense(n_second_layer, activation='relu', kernel_constraint=max_norm(c)))
    model.add(Dropout(prob_dropout))
    
    # put out final prediction
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer=opt, loss=MeanSquaredLogarithmicError(), metrics=['mean_absolute_error'])
    return model



####  set of functions for bayesian neural net

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model
def get_bayesian_model(input_dim, base_n_nodes, multiplier_n_nodes, posterior, prior, lr, train_size):
    
    # define optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # define how many in second layer
    n_second_layer = int(round(base_n_nodes* multiplier_n_nodes))
    
    # the number in each layer
    hidden_units = [base_n_nodes, n_second_layer]
    
    
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))

        # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="relu",
        )(features)
    
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer=opt, loss=MeanSquaredLogarithmicError(), metrics=['mean_absolute_error'])
    return model
    
    