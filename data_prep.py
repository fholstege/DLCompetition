"""
DATA PREPARATION

input: dropbox data files on housing data
output: cleaned train and test data for training (validation set still has to be taken)
"""
import pandas as pd
from sklearn import preprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import os 


# load train and test dataframes
df_train = pd.read_csv("https://www.dropbox.com/s/bawlkeolef1bse2/train_dat.csv?dl=1", 

                        sep= ",")
df_test = pd.read_csv("https://www.dropbox.com/s/rbjatpuk5x7dios/test_dat.csv?dl=1", 
                         sep= ",")

# ensure correct variable types and encoding
final_dfs = [df_train, df_test]

################################################
# COMMENT AND UNCOMMENT BASED ON USE CASE
################################################

# ORIGINAL CASE
# leave yearbuilt etc as categorical, add no variables
# determine categorical features
#l_cat_features = ["YearBuilt", "YearRemodAdd", "MoSold", "YrSold"]

#for i, df in enumerate(final_dfs):
    # get categorical variables coded as dummies
    # do NOT drop first dummy, let NN figure out what is important
#    final_dfs[i] = pd.get_dummies(df, columns = l_cat_features)

# CASE 1
# leave yearbuilt etc as categorical, make quality variables categorical
# alternative categorical features
#l_cat_features = ["YearBuilt", "YearRemodAdd", "MoSold", "YrSold", "OverallQual", "OverallCond"]

#for i, df in enumerate(final_dfs):
     # get categorical variables coded as dummies
     # do NOT drop first dummy, let NN figure out what is important
#     final_dfs[i] = pd.get_dummies(df, columns = l_cat_features)

# CASE 2
# leave all categorical, add qual variables as categorical, add additional variables
# l_cat_features = ["YearBuilt", "YearRemodAdd", "MoSold", "YrSold", "OverallQual", "OverallCond"]

# for i, df in enumerate(final_dfs):
    
#     # construct additional features
#     # indicators
#     # openporch, wooddeck, enclosedporch, screenporch
#     final_dfs[i]["HasOpenPorch"] = pd.get_dummies(final_dfs[i]["OpenPorchSF"] > 0, drop_first=True)

#     final_dfs[i]["HasWoodDeck"] = pd.get_dummies(final_dfs[i]["WoodDeckSF"] > 0, drop_first=True)

#     final_dfs[i]["HasEnclPorch"] = pd.get_dummies(final_dfs[i]["EnclosedPorch"] > 0, drop_first=True)

#     final_dfs[i]["HasScreenPorch"] = pd.get_dummies(final_dfs[i]["ScreenPorch"] > 0, drop_first=True)
    
#     # adding total square feed and time differences
#     final_dfs[i]["RemodMinusBuilt"] =  final_dfs[i]["YearRemodAdd"] - final_dfs[i]["YearBuilt"] + 1
#     final_dfs[i]["SoldMinusRemod"] = final_dfs[i]["YrSold"] -  final_dfs[i]["YearRemodAdd"] + 1
#     final_dfs[i]["TotalHouseSF"] = final_dfs[i]["TotalBsmtSF"] + final_dfs[i]["X1stFlrSF"] + final_dfs[i]["X2ndFlrSF"]
    
#     # get categorical variables coded as dummies
#     # do NOT drop first dummy, let NN figure out what is important
#     final_dfs[i] = pd.get_dummies(df, columns = l_cat_features)


# CASE 3
# years and quality as continuous variables, add variables

for i, df in enumerate(final_dfs):
    
     # construct additional features
     # indicators
     # openporch, wooddeck, enclosedporch, screenporch
     final_dfs[i]["HasOpenPorch"] = pd.get_dummies(final_dfs[i]["OpenPorchSF"] > 0, drop_first=True)

     final_dfs[i]["HasWoodDeck"] = pd.get_dummies(final_dfs[i]["WoodDeckSF"] > 0, drop_first=True)

     final_dfs[i]["HasEnclPorch"] = pd.get_dummies(final_dfs[i]["EnclosedPorch"] > 0, drop_first=True)

     final_dfs[i]["HasScreenPorch"] = pd.get_dummies(final_dfs[i]["ScreenPorch"] > 0, drop_first=True)
    
     # adding total square feed and time differences
     final_dfs[i]["RemodMinusBuilt"] =  final_dfs[i]["YearRemodAdd"] - final_dfs[i]["YearBuilt"] + 1
     final_dfs[i]["SoldMinusRemod"] = final_dfs[i]["YrSold"] -  final_dfs[i]["YearRemodAdd"] + 1
     final_dfs[i]["TotalHouseSF"] = final_dfs[i]["TotalBsmtSF"] + final_dfs[i]["X1stFlrSF"] + final_dfs[i]["X2ndFlrSF"]
    

################################################
       
# unpack final datasets
df_train, df_test = final_dfs

# which categories / vars do not exist in test data
list(set(df_train.columns) - set(df_test.columns))

# construct file path
data_path = Path(os.getcwd()).parent/"DLCompetition/data"

# save as hdf5
df_train.to_hdf(data_path/"train_df.hdf5", mode = "w", key = "train")
df_test.to_hdf(data_path/"test_df.hdf5", mode = "w", key = "test")