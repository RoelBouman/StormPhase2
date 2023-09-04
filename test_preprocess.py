#%% Load packages
import os

import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt


from src.methods import SingleThresholdStatisticalProfiling
from src.preprocess import preprocess_per_batch_and_write
from src.io_functions import save_dataframe_list, load_batch

from src.preprocess import match_bottomup_load


#%% load data

data_folder = "data"

which_split = "Train"
X_train_dfs, y_train_dfs, X_train_files= load_batch(data_folder, which_split) 

preprocessing_hyperparameters = {'subsequent_nr': 5, 'lin_fit_quantiles': (10, 90)}


#%% def functions

def test_preprocess(X_df: pd.DataFrame, y_df: pd.DataFrame, subsequent_nr: int, lin_fit_quantiles: tuple) -> pd.DataFrame:
    """Match bottom up with substation measurements with linear regression and apply the sign value to the substation measurements.

    Args:
        df (pd.DataFrame): Dataframe with at least the columns M_TIMESTAMP, S_original, BU_original and Flag.
        subsequent_nr (int): Integer that represents the number of subsequent equal measurements
        line_fit_quantiles (tuple): A tuple containing the lower and upper quantiles for the linear fit model

    Returns:
        pd.DataFrame: DataFrame with the columns M_TIMESTAMP, S_original, BU_original, diff_original, S, BU, diff, and missing.
    """ 
    # Calculate difference and add label column.
    X_df['diff_original'] = X_df['S_original']-X_df['BU_original']
        
    # Flag measurement mistakes BU and SO
    # 0 okay
    # 1 measurement missing
    # 2 bottom up missing
    
    X_df['S'] = X_df['S_original'].copy()
    X_df['missing'] = 0
    X_df.loc[X_df['BU_original'].isnull(),'missing'] = 2
    
    prev_v = 0
    prev_i = 0
    count = subsequent_nr
    
    # Flag measurement as missing when # of times after each other the same value
    for i, v in enumerate(X_df['S']):
        # if value is same as previous, decrease count by 1
        if v == prev_v:
            count -= 1
            continue
            
        # if not, check if previous count below zero, if so, set all missing values to 1
        elif count <= 0:
            X_df.loc[prev_i:i - 1, 'missing'] = 1
            
        # reset vars
        prev_v = v
        prev_i = i
        count = subsequent_nr
    
    # Match bottom up with substation measurements for the middle 80% of the values and apply sign to substation measurements
    arr = X_df[X_df['missing']==0]
    
    low_quant, up_quant = lin_fit_quantiles
    low_quant_value = np.percentile(arr['diff_original'],low_quant)
    up_quant_value = np.percentile(arr['diff_original'],up_quant)
    
    arr = arr[np.logical_and(arr['diff_original'] > low_quant_value, arr['diff_original'] < up_quant_value)]
    
    a, b = match_bottomup_load(bottomup_load=arr['BU_original'], measurements=arr['S_original'])
    X_df['BU'] = a*X_df['BU_original']+b
    if X_df['S_original'].min()>0:
        X_df['S'] = np.sign(X_df['BU'])*X_df['S']
    X_df['diff'] = X_df['S']-X_df['BU']
        
    # remove all diff_original NaN in X, adjust y correspondingly
    y_df = y_df[X_df['diff_original'].notna()]
    X_df = X_df[X_df['diff_original'].notna()]
    
    # reset df index
    y_df = y_df.reset_index()
    X_df = X_df.reset_index()
    
    return X_df[['M_TIMESTAMP', 
               'S_original', 'BU_original', 'diff_original', 
               'S', 'BU', 'diff', 
               'missing']], y_df


def data_to_score(df, bkps):
    y_score = np.zeros(len(df))
    total_mean = np.mean(df) # calculate mean of all values in timeseries
    
    prev_bkp = 0
        
    for bkp in bkps:
        segment = df[prev_bkp:bkp] # define a segment between two breakpoints
        segment_mean = np.mean(segment)
        
        # for all values in segment, set its score to th difference between the total mean and the mean of the segment its in
        y_score[prev_bkp:bkp] = total_mean - segment_mean   
        
        prev_bkp = bkp
    
    return y_score   

#%% 

dfs_preprocessed = [test_preprocess(X_df, y_df, **preprocessing_hyperparameters) for (X_df, y_df) in zip(X_train_dfs, y_train_dfs)]

X_dfs_preprocessed = [X_df for (X_df, y_df) in dfs_preprocessed]
y_dfs_preprocessed = [y_df for (X_df, y_df) in dfs_preprocessed] 


#%%

"""
# check which files have NaN values

for i, df in enumerate(X_train_dfs):
    if df["BU_original"].isna().sum() != 0:
        print(X_train_files[i])
"""


df = X_train_dfs[48]
signal = np.array(df['S_original']-df['BU_original'])

n = len(signal) # nr of samples
sigma = np.std(signal) * 3 #noise standard deviation

# hyperparameters
model = "l1"
min_size = 100
jump = 10
penalty = np.log(n) * sigma**2

# detection
algo = rpt.Binseg(model=model, min_size=min_size, jump=jump)
result = algo.fit_predict(signal, pen = penalty)

# display
rpt.display(signal, result)
plt.show()



    
    