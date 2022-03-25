#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""

import os
import pickle

import pandas as pd
import numpy as np

from scipy.optimize import minimize

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler

from src.evaluation import STORM_score
from src.evaluation import threshold_scores
from src.evaluation import double_threshold_scores
from src.evaluation import inv_threshold_and_score
from src.evaluation import inv_double_threshold_and_score

cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

data_folder = "data"
result_folder = "results"

pickle_train_file_folder = os.path.join(data_folder, "X_train_preprocessed_pickle")
pickle_test_file_folder = os.path.join(data_folder, "X_test_preprocessed_pickle")

#%% define station data getter function

def get_all_station_data(data_name, result_folder, method_name, hyperparameter_string, pickle_folder):
    
    station_score_folder = os.path.join(result_folder, data_name, method_name, hyperparameter_string)
    X_file_folder = pickle_folder
    
    station_score_files = os.listdir(station_score_folder)
    
    X_files = os.listdir(X_file_folder)
    
    if station_score_files == X_files:
        #Current method of appending is fast as long as memory is not limiting, if memory is limiting, matches should be calculated per station, and recombined afterwards.
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
        event_lengths_combined = np.concatenate(event_lengths)
        
        #remove samples where y_true == 5
        
        y_scores_filtered = y_scores_combined[y_true_combined != 5]
        y_true_filtered = y_true_combined[y_true_combined != 5]
        event_lengths_filtered = event_lengths_combined[y_true_combined != 5]
        
        return (y_scores_filtered, y_true_filtered, event_lengths_filtered)
    else:
        Warning("Not all scores for this hyperparameter setting have been calculated yet. Skipping...")

#%% Isolation Forest score calculation


print("Now calculating:")
print("Isolation Forest")



#Perform grid search
#calculate combinations from hyperparameters[method_name]
hyperparameter_grid = {"n_estimators":[1000], "bootstrap":[True, False], "max_samples":[128,256,512,1024]}

hyperparameter_list = list(ParameterGrid(hyperparameter_grid))


def get_IF_scores(pickle_folder, data_name, hyperparameter_list):
    for pickle_file in os.listdir(pickle_folder):
        method_name = "IF"
        substation_name = pickle_file[:-7]
        
        print("Station: ")
        print(substation_name)
        
        data = pickle.load(open(os.path.join(pickle_folder, pickle_file), 'rb'))
        X, y = data["X"].reshape(-1,1), np.squeeze(data["y"].reshape(-1,1))
        
        for hyperparameter_settings in hyperparameter_list:
            #hyperparameter_string = re.sub(r"[^a-zA-Z0-9_ ]","",str(hyperparameter_settings))
            hyperparameter_string = str(hyperparameter_settings)
            
            result_file_path = os.path.join(result_folder, data_name, method_name, hyperparameter_string, substation_name+".pickle")
            
            if os.path.exists(result_file_path):
                pass
            else:        
                print("evaluating hyperparameter setting:")
                print(hyperparameter_settings)
                #evaluate method using hyperparameter_settings
                
                model = IsolationForest(**hyperparameter_settings)
                model.fit(X)
                y_scores = model.decision_function(X)
                
                if not os.path.exists(os.path.join(result_folder, data_name, method_name, hyperparameter_string)):
                    os.makedirs(os.path.join(result_folder, data_name, method_name, hyperparameter_string))
                
                with open(result_file_path, 'wb') as handle:
                    pickle.dump(y_scores, handle)

print("Calculating train scores:")
get_IF_scores(pickle_train_file_folder, data_name="X_train", hyperparameter_list=hyperparameter_list)

#%% Isolation Forest score evaluation
print("Now evaluating:")
print("Isolation Forest")

method_name="IF"

best_score = 0
best_hyperparameters= []
best_threshold = 0

print("Evaluate training data:")
for hyperparameter_settings in hyperparameter_list:
    hyperparameter_string = str(hyperparameter_settings)
    print(hyperparameter_string)
    
    # check if all results have actually been calculated
    y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_train", result_folder, method_name, hyperparameter_string, pickle_train_file_folder)

    res = minimize(inv_threshold_and_score, 0, args=(y_true_filtered, y_scores_filtered, event_lengths_filtered, cutoffs), method="Nelder-Mead", options ={"disp":False})
    
    threshold = res.x[0]
    
    y_pred = threshold_scores(y_scores_filtered, threshold)
    
    storm_score = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs)
    print("Best STORM score:")
    print(storm_score)
    print("threshold:")
    print(threshold)
    
    if storm_score > best_score:
        best_score = storm_score
        best_hyperparameters = hyperparameter_settings
        best_threshold = threshold
        
#evaluate on test data:
print("Evaluate test data:")
get_IF_scores(pickle_test_file_folder, "X_test", [best_hyperparameters])
hyperparameter_string = str(best_hyperparameters)

y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_test", result_folder, method_name, hyperparameter_string, pickle_test_file_folder)

y_pred = threshold_scores(y_scores_filtered, best_threshold)

storm_score = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs)

print("STORM score on test:")
print(storm_score)

#%%Robust interval calculation:


print("Now calculating:")
print("Robust interval")

def get_RI_scores(pickle_folder, data_name, hyperparameter_list):
    for pickle_file in os.listdir(pickle_folder):
        method_name = "RI"
        substation_name = pickle_file[:-7]
        
        print("Station: ")
        print(substation_name)
        
        data = pickle.load(open(os.path.join(pickle_folder, pickle_file), 'rb'))
        X, y = data["X"].reshape(-1,1), np.squeeze(data["y"].reshape(-1,1))
        
        for hyperparameter_settings in hyperparameter_list:
            #hyperparameter_string = re.sub(r"[^a-zA-Z0-9_ ]","",str(hyperparameter_settings))
            hyperparameter_string = str(hyperparameter_settings)
            
            result_file_path = os.path.join(result_folder, data_name, method_name, hyperparameter_string, substation_name+".pickle")
            
            if os.path.exists(result_file_path):
                pass
            else:        
                print("evaluating hyperparameter setting:")
                print(hyperparameter_settings)
                #evaluate method using hyperparameter_settings
                
                model = RobustScaler(**hyperparameter_settings)
                
                y_scores = np.squeeze(model.fit_transform(X))
                
                if not os.path.exists(os.path.join(result_folder, data_name, method_name, hyperparameter_string)):
                    os.makedirs(os.path.join(result_folder, data_name, method_name, hyperparameter_string))
                
                with open(result_file_path, 'wb') as handle:
                    pickle.dump(y_scores, handle)


#Perform grid search
#calculate combinations from hyperparameters[method_name]
hyperparameter_grid = {"quantile_range":[(10,90), (5, 95), (2.5, 97.5), (15,85), (20,80),(25,75)]}

hyperparameter_list = list(ParameterGrid(hyperparameter_grid))

get_RI_scores(pickle_train_file_folder, data_name="X_train", hyperparameter_list=hyperparameter_list)

#%% evaluate Robust interval
method_name="RI"

best_score = 0
best_hyperparameters= []
best_thresholds = 0


print("Evaluate training data:")
for hyperparameter_settings in hyperparameter_list:
    hyperparameter_string = str(hyperparameter_settings)
    print(hyperparameter_string)
    
    # check if all results have actually been calculated
    y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_train", result_folder, method_name, hyperparameter_string, pickle_train_file_folder)

    res = minimize(inv_double_threshold_and_score, (-1,1), args=(y_true_filtered, y_scores_filtered, event_lengths_filtered, cutoffs), method="Nelder-Mead", options ={"disp":False})
    
    thresholds = res.x
    
    y_pred = double_threshold_scores(y_scores_filtered, thresholds)
    
    storm_score = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs)
    print("Best STORM score:")
    print(storm_score)
    print("thresholds:")
    print(thresholds)
    
    if storm_score > best_score:
        best_score = storm_score
        best_hyperparameters = hyperparameter_settings
        best_thresholds = thresholds
        
#evaluate on test data:
print("Evaluate test data:")
get_RI_scores(pickle_test_file_folder, "X_test", [best_hyperparameters])

y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_test", result_folder, method_name, hyperparameter_string, pickle_test_file_folder)

y_pred = double_threshold_scores(y_scores_filtered, best_thresholds)

storm_score = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs)

print("STORM score on test:")
print(storm_score)