#%% Load packages
import os

import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

from src.methods import SingleThresholdStatisticalProfiling
from src.preprocess import preprocess_data
from src.io_functions import save_dataframe_list, load_batch

from src.preprocess import match_bottomup_load


#%% load data

data_folder = "data"

which_split = "Train"
X_train_dfs, y_train_dfs, X_train_files= load_batch(data_folder, which_split) 

preprocessing_hyperparameters = {'subsequent_nr': 5, 'lin_fit_quantiles': (10, 90)}


#%% 

dfs_preprocessed = [preprocess_data(X_df, y_df, **preprocessing_hyperparameters) for (X_df, y_df) in zip(X_train_dfs, y_train_dfs)]

X_dfs_preprocessed = [X_df for (X_df, y_df) in dfs_preprocessed]
y_dfs_preprocessed = [y_df for (X_df, y_df) in dfs_preprocessed]

#%%

def fused_lasso(signal, beta):
    tot_sum = 0
    mean = np.mean(signal)
    for i in signal:
        tot_sum += np.abs(i - mean)
    
    return beta * tot_sum

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

#scores = []

for i in range(65):
    df = X_dfs_preprocessed[i]
    signal = df['diff'].values.reshape(-1,1)
    
    scaler = RobustScaler(quantile_range= (5,95))
    
    signal = scaler.fit_transform(signal)
    
    n = len(signal) # nr of samples
    sigma = np.std(signal) * 3 #noise standard deviation
    beta = 0.01
    
    # hyperparameters
    model = "l1"
    min_size = 100
    jump = 10
    
    #penalty = beta * n # linear penalty
    #penalty = 1/2 * np.log(n) * n # BIC pen
    #penalty = np.log(n) * sigma**2 # BIC_L2 pen 
    #penalty = sigma**2 * n # AIC penalty
    penalty = fused_lasso(signal, beta)
    
    # detection
    algo = rpt.Binseg(model=model, min_size=min_size, jump=jump)
    result = algo.fit_predict(signal, pen = penalty)
    
    #scores.append(data_to_score(signal, result))
    
    # display
    rpt.display(signal, result)
    plt.savefig("binseg_plots/binseg_" + str(i))
    plt.show()



    
    