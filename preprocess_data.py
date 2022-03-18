#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""

import os

import pandas as pd
import numpy as np

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
            
            # Add column with the length of the switching events expressed in nr of datapoints
            length = np.zeros(len(y_df))
            labels = list(y_df['label'] != 0) + [False]
            end_inx = 0

            for _ in range(len(y_df)):
                try:
                    start_inx = labels[end_inx:].index(True) + end_inx
                    end_inx = labels[start_inx:].index(False) + start_inx
                    length[start_inx:end_inx] = end_inx - start_inx
                except:
                    break
            X_df_preprocessed['length'] = length
            
            #write full results
            target_path_folder = os.path.join(data_folder, "X_"+subset+"_preprocessed_full")
            if not os.path.exists(target_path_folder):
                os.makedirs(target_path_folder)
            
            target_path = os.path.join(target_path_folder, X_csv_file)
            
            X_df_preprocessed.to_csv(target_path)
            
            
            
            #write minimal X, y, lengths pickle
            
            X = X_df_preprocessed["diff"].values[X_df_preprocessed['diff_original'].notnull()]
            
            y = y_df["label"].values[X_df_preprocessed['diff_original'].notnull()]
            
            lengths = X = X_df_preprocessed["length"].values[X_df_preprocessed['diff_original'].notnull()]
            
            data_dict = {"X":X , "y":y, "lengths":lengths}
            
            target_path_folder = os.path.join(data_folder, "X_"+subset+"_preprocessed_pickle")
            if not os.path.exists(target_path_folder):
                os.makedirs(target_path_folder)
            
            target_path = os.path.join(target_path_folder, substation_name+".pickle",)
            pickle.dump(data_dict, open(target_path, "wb"))    