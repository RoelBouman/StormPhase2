#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:35:25 2024

@author: rbouman
"""

import os
from sklearn.model_selection import ParameterGrid
from hashlib import sha256
import numpy as np
import matplotlib.pyplot as plt

from src.io_functions import load_batch
from src.preprocess import preprocess_per_batch_and_write

from src.plot_functions import plot_S_original, plot_BU_original

raw_data_folder = "raw_data"
dataset = "route_data"
intermediates_folder = os.path.join(raw_data_folder, dataset+"_preprocessed")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]



#%% 
X_dfs, y_dfs, X_files = load_batch(raw_data_folder, dataset)


import matplotlib.style as mplstyle
mplstyle.use('fast')

#%%
for y_df, X_df, name in zip(y_dfs, X_dfs, X_files):
    plt.figure()
    plt.subplot(4,1,1)
    plt.title(name)
    X_df["S_original"] = X_df["S_original"]*1000
    plot_S_original(X_df, label="S original")
    plt.subplot(4,1,2)
    plot_BU_original(X_df, label="BU original")
    plt.subplot(4,1,3)
    plt.plot(X_df["missing"])
    plt.subplot(4,1,4)
    plt.plot(y_df["label"])
    
    plt.show()