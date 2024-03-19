#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:14:41 2023

@author: rbouman
"""

import os
import pandas as pd
import pickle

def save_dataframe_list(dfs, station_names, folder, overwrite):
    for df, station_name in zip(dfs, station_names):
        
        file_name = os.path.join(folder, station_name)
        
        os.makedirs(folder, exist_ok = True)
        if overwrite or not os.path.exists(file_name):
            
            df.to_csv(file_name)
            
def load_dataframe_list(data_folder):
    data_path = os.path.join(data_folder, "stations")
    
    data_files = sorted(os.listdir(data_path))
    
    dfs = []
    for file in data_files:
        dfs.append(pd.read_csv(os.path.join(data_path, file)))
        
    return dfs, data_files
            
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
    
def save_metric(metric, metric_path, hyperparameter_string):
    os.makedirs(metric_path, exist_ok=True)
    metric_df = pd.DataFrame([metric])
    
    metric_df.to_csv(os.path.join(metric_path, hyperparameter_string+".csv"), index=False, header=False)

def load_metric(metric_path, hyperparameter_string):
    metric_df = pd.read_csv(os.path.join(metric_path, hyperparameter_string+".csv"), header=None)
    
    metric = metric_df.iloc[0,0]
    
    return metric

def save_table(table, metric_path, hyperparameter_string):
    os.makedirs(metric_path, exist_ok=True)
    
    table.to_csv(os.path.join(metric_path, hyperparameter_string+".csv"), index=True, header=True)

def load_table(table_path, hyperparameter_string):
    table = pd.read_csv(os.path.join(table_path, hyperparameter_string+".csv"))
    table.set_index("Cutoffs", inplace=True)

    return table

def save_minmax_stats(minmax_stats, metric_path, hyperparameter_string):
    os.makedirs(metric_path, exist_ok=True)    

    minmax_stats.to_csv(os.path.join(metric_path, hyperparameter_string+".csv"), index=False, header=True)
    
def load_minmax_stats(minmax_stats_path, hyperparameter_string):
    stats_df = pd.read_csv(os.path.join(minmax_stats_path, hyperparameter_string+".csv"))

    return stats_df


def print_count_nan(df, column=None):
    if column == None:
        total_nan = df.isna().sum()
    else:
        total_nan = df[column].isna().sum()
        
    print(total_nan)