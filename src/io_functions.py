#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:14:41 2023

@author: rbouman
"""

import os
import pandas as pd

def save_dataframe_list(dfs, station_names, folder, overwrite):
    for df, station_name in zip(dfs, station_names):
        
        file_name = os.path.join(folder, station_name)
        
        os.makedirs(folder, exist_ok = True)
        if overwrite or not os.path.exists(file_name):
            

            df.to_csv(file_name)
            
def load_batch(data_folder, which_split):
    X_path = os.path.join(data_folder, which_split, "X")
    y_path = os.path.join(data_folder, which_split, "y")

    X_files = sorted(os.listdir(X_path))
    y_files = sorted(os.listdir(y_path))

    if not X_files == y_files:
        raise RuntimeError("Not all training files are present in both the X and y folders.")

    file_names = X_files

    X_dfs, y_dfs = [], []
    for file in file_names:
        X_dfs.append(pd.read_csv(os.path.join(X_path, file)))
        y_dfs.append(pd.read_csv(os.path.join(y_path, file)))
        
    return X_dfs, y_dfs, file_names