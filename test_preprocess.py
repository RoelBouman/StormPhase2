#%% Load packages
import os

import numpy as np
import pandas as pd

from src.methods import SingleThresholdStatisticalProfiling
from src.preprocess import preprocess_per_batch_and_write
from src.io_functions import save_dataframe_list, load_batch

#%% load data

data_folder = "data"

which_split = "Train"
X_train_dfs, y_train_dfs, X_train_files= load_batch(data_folder, which_split)

#%% def functions

def test_clock_moving(df_X: pd.DataFrame) -> pd.DataFrame:
    print(df_X.loc['Flag'])
    try:
        print("Test: " + df_X.loc[(df_X['M_TIMESTAMP']>='2020-03-29 03:00:00')
              &(df_X['M_TIMESTAMP']< '2020-03-29 04:00:00'),'Flag'])
        df_X.loc[(df_X['M_TIMESTAMP']>='2020-03-29 03:00:00')
              &(df_X['M_TIMESTAMP']< '2020-03-29 04:00:00'),'Flag'] = "5"
        print(df_X.loc[(df_X['M_TIMESTAMP']>='2020-03-29 03:00:00')
              &(df_X['M_TIMESTAMP']< '2020-03-29 04:00:00'),'Flag'])
        
        df_X.loc[(df_X['M_TIMESTAMP']>='2020-03-29 03:00:00')
              &(df_X['M_TIMESTAMP']< '2020-03-29 04:00:00'),'S_original'] = list(df_X[df_X['M_TIMESTAMP']<'2020-03-29 00:03:00']['S_original'])[-1]
    except:
        pass
    
    return df_X

#%% pre_process

X_dfs_preprocessed = [test_clock_moving(df) for df in X_train_dfs]



    
    