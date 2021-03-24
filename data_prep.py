"""
DATA PREPARATION

input: dropbox data files on housing data
output: cleaned train and test data for training (validation set still has to be taken)
"""
import pandas as pd
from sklearn import preprocessing
from pathlib import Path

# load train and test dataframes
df_train = pd.read_csv("https://www.dropbox.com/s/bawlkeolef1bse2/train_dat.csv?dl=1", 

                        sep= ",")
df_test = pd.read_csv("https://www.dropbox.com/s/rbjatpuk5x7dios/test_dat.csv?dl=1", 
                         sep= ",")

# determine categorical features
l_cat_features = ["YearBuilt", "YearRemodAdd", "MoSold", "YrSold"]

# ensure correct variable types and encoding
final_dfs = [df_train, df_test]
for i, df in enumerate(final_dfs):
    
    # get categorical variables coded as dummies
    # do NOT drop first dummy, let NN figure out what is important
    final_dfs[i] = pd.get_dummies(df, columns = l_cat_features)
    
# unpack
df_train, df_test = final_dfs

# construct file path
data_path = (Path(__file__).parent)

# save as hdf5
df_train.to_hdf(data_path/"data/train_df.hdf5", mode = "w", key = "train")
df_test.to_hdf(data_path/"data/test_df.hdf5", mode = "w", key = "test")