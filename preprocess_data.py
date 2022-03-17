#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""

import os

import pandas as pd

from src.preprocess import preprocess_data

import pickle

data_folder = "data"

output_folder = "results"

#%% Preprocess data 
subsets = ["train", "test"]

for subset in subsets:
    print("preprocessing:")
    print(subset)
    
    X_dir = os.path.join(data_folder, "X_"+subset)
    y_dir = os.path.join(data_folder, "y_"+subset)
    
    X_csv_files = [file for file in os.listdir(X_dir) if file.endswith('.csv')]
    y_csv_files = [file for file in os.listdir(y_dir) if file.endswith('.csv')]
    

    for X_csv_file, y_csv_file in zip(X_csv_files, y_csv_files):
            substation_name = X_csv_file[:-4]
            
            print("Station: ")
            print(substation_name)
            
            X_df = pd.read_csv(os.path.join(X_dir, X_csv_file))
            y_df = pd.read_csv(os.path.join(y_dir, y_csv_file))
            
            X_df_preprocessed = preprocess_data(X_df)
            
            #write full results
            target_path_folder = os.path.join(data_folder, "X_"+subset+"_preprocessed_full")
            if not os.path.exists(target_path_folder):
                os.makedirs(target_path_folder)
            
            target_path = os.path.join(target_path_folder, X_csv_file)
            
            X_df_preprocessed.to_csv(target_path)
            
            #write minimal X, y pickle
            
            X = X_df_preprocessed["diff"].values[X_df_preprocessed['diff_original'].notnull()]
            
            y = y_df["label"].values[X_df_preprocessed['diff_original'].notnull()]
            data_dict = {"X":X , "y":y}
            
            target_path_folder = os.path.join(data_folder, "X_"+subset+"_preprocessed_pickle")
            if not os.path.exists(target_path_folder):
                os.makedirs(target_path_folder)
            
            target_path = os.path.join(target_path_folder, substation_name+".pickle",)
            pickle.dump(data_dict, open(target_path, "wb"))    