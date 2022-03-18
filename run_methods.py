#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""

import os
import pickle
import re

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import fbeta_score

from src.isolation_forest import threshold_scores


data_folder = "data"
result_folder = "results"

pickle_train_file_folder = os.path.join(data_folder, "X_train_preprocessed_pickle")
pickle_test_file_folder = os.path.join(data_folder, "X_test_preprocessed_pickle")

#%% Isolation Forest score calculation


print("Now calculating:")
print("Isolation Forest")

method_name = "IF"

#Perform grid search
#calculate combinations from hyperparameters[method_name]
hyperparameter_grid = {"n_estimators":[100,1000]}

hyperparameter_list = list(ParameterGrid(hyperparameter_grid))


for pickle_file in os.listdir(pickle_train_file_folder):
    
    substation_name = pickle_file[:-7]
    
    print("Station: ")
    print(substation_name)
    
    data = pickle.load(open(os.path.join(pickle_train_file_folder, pickle_file), 'rb'))
    X, y = data["X"].reshape(-1,1), np.squeeze(data["y"].reshape(-1,1))
    
    for hyperparameter_settings in hyperparameter_list:
        #hyperparameter_string = re.sub(r"[^a-zA-Z0-9_ ]","",str(hyperparameter_settings))
        hyperparameter_string = str(hyperparameter_settings)
        
        result_file_path = os.path.join(result_folder, "X_train", method_name, hyperparameter_string, substation_name+".pickle")
        
        if os.path.exists(result_file_path):
            pass
        else:        
            print("evaluating hyperparameter setting:")
            print(hyperparameter_settings)
            #evaluate method using hyperparameter_settings
            
            model = IsolationForest(**hyperparameter_settings)
            model.fit(X)
            y_scores = model.decision_function(X)
            
            if not os.path.exists(os.path.join(result_folder, "X_train", method_name, hyperparameter_string)):
                os.makedirs(os.path.join(result_folder, "X_train", method_name, hyperparameter_string))
            
            with open(result_file_path, 'wb') as handle:
                pickle.dump(y_scores, handle)
            

#%% Isolation Forest score evaluation
print("Now evaluating:")
print("Isolation Forest")

method_name = "IF"

for hyperparameter_settings in hyperparameter_list:
    hyperparameter_string = str(hyperparameter_settings)
    
    # check if all results have actually been calculated
    
    station_score_folder = os.path.join(result_folder, "X_train", method_name, hyperparameter_string)
    X_file_folder = os.path.join(data_folder, "X_train_preprocessed_pickle")
    
    station_score_files = os.listdir(station_score_folder)
    
    X_files = os.listdir(X_file_folder)
    
    if station_score_files == X_files:
        y_scores = []
        y_true = []
        
        event_lengths = []
        
        for station_score_file in station_score_files:
            
            y_scores.append(pickle.load(open(os.path.join(station_score_folder, station_score_file), 'rb')))
            #y_score = pickle.load(open(os.path.join(station_score_folder, station_score_file), 'rb'))
            data = pickle.load(open(os.path.join(X_file_folder, station_score_file), 'rb'))
            
            y_true.append(data["y"])
            
            event_lengths.append(data["lengths"])
            
        y_scores_combined = np.concatenate(y_scores)
        y_true_combined = np.concatenate(y_true)
        
        #remove samples where y_true == 5
        
        y_scores_filtered = y_scores_combined[y_true_combined != 5]
        y_true_filtered = y_true_combined[y_true_combined != 5]
        
        #Write evaluation function working per length
        
        #Should we handle suspect missing data at evaluation or calculation time. Linda did it at evaluation, but this might lead to problems with time series methods such as binseg
           
    else:
        Warning("Not all scores for this hyperparameter setting have been calculated yet. Skipping...")
        