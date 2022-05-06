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
from scipy import optimize

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler

from src.evaluation import STORM_score
from src.evaluation import threshold_scores
from src.evaluation import double_threshold_scores
from src.evaluation import neg_threshold_and_score
from src.evaluation import neg_double_threshold_and_score

from src.helper_functions import find_BS_thresholds

import rpy2.robjects.packages as rpackages
from rpy2.robjects import r, pandas2ri

changepoint = rpackages.importr('changepoint')


all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]
low_cutoffs = [(0, 24), (24, 288)]
high_cutoffs = [(288, 4032), (4032, np.inf)]
low_and_all_cutoffs = [low_cutoffs, all_cutoffs]
high_and_all_cutoffs = [high_cutoffs, all_cutoffs]

data_folder = "data"
prediction_folder = "predictions"
intermediate_folder = "intermediates"
result_folder = "results"
thresholds_folder = "thresholds"

pickle_train_file_folder = os.path.join(data_folder, "X_train_preprocessed_pickle")
pickle_test_file_folder = os.path.join(data_folder, "X_test_preprocessed_pickle")

#%% define station data getter function


def get_y_true_and_lengths(pickle_folder, filter_labels=False):
    
    X_files = os.listdir(pickle_folder)
    
    y_true = []
    event_lengths = []
    
    for X_file in X_files:
        
        data = pickle.load(open(os.path.join(pickle_folder, X_file), 'rb'))
        
        y_true.append(data["y"])
        
        event_lengths.append(data["lengths"])
        

    y_true_combined = np.concatenate(y_true)
    event_lengths_combined = np.concatenate(event_lengths)
    
    if filter_labels:
        y_true_filtered = y_true_combined[y_true_combined != 5]
        event_lengths_filtered = event_lengths_combined[y_true_combined != 5]
        
        return (y_true_filtered, event_lengths_filtered)
    else:
        return (y_true_combined, event_lengths_combined)

def get_all_station_data(data_name, prediction_folder, method_name, hyperparameter_string, pickle_folder, get_scores=True):
    
    station_score_folder = os.path.join(prediction_folder, data_name, method_name, hyperparameter_string)
    
    station_score_files = os.listdir(station_score_folder)
    
    X_files = os.listdir(pickle_folder)
    
    if station_score_files == X_files:
        #Current method of appending is fast as long as memory is not limiting, if memory is limiting, matches should be calculated per station, and recombined afterwards.
        
        y_true_combined, event_lengths_combined = get_y_true_and_lengths(pickle_folder)
        
        y_scores = []
        
        for station_score_file in station_score_files:
            
            y_scores.append(pickle.load(open(os.path.join(station_score_folder, station_score_file), 'rb')))
            
        y_scores_combined = np.concatenate(y_scores)
        y_scores_filtered = y_scores_combined[y_true_combined != 5]
        
        #remove samples where y_true == 5
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
            
            prediction_file_path = os.path.join(prediction_folder, data_name, method_name, hyperparameter_string, substation_name+".pickle")
            
            if os.path.exists(prediction_file_path):
                pass
            else:        
                print("evaluating hyperparameter setting:")
                print(hyperparameter_settings)
                #evaluate method using hyperparameter_settings
                
                model = IsolationForest(**hyperparameter_settings)
                model.fit(X)
                y_scores = model.decision_function(X)
                
                if not os.path.exists(os.path.join(prediction_folder, data_name, method_name, hyperparameter_string)):
                    os.makedirs(os.path.join(prediction_folder, data_name, method_name, hyperparameter_string))
                
                with open(prediction_file_path, 'wb') as handle:
                    pickle.dump(y_scores, handle)

print("Calculating train scores:")
get_IF_scores(pickle_train_file_folder, data_name="X_train", hyperparameter_list=hyperparameter_list)

#%% Isolation Forest score evaluation
print("Now evaluating:")
print("Isolation Forest")

method_name="IF"

for cutoffs in low_and_all_cutoffs:
    cutoffs_string = str(cutoffs)
    print("Evaluating cutoffs:" + cutoffs_string)
    
    best_score = 0
    best_hyperparameters= []
    best_threshold = 0
    
    print("Evaluate training data:")
    for hyperparameter_settings in hyperparameter_list:
        hyperparameter_string = str(hyperparameter_settings)
        print(hyperparameter_string)
        
        
        result_file_path = os.path.join(result_folder, cutoffs_string, "X_train", method_name, hyperparameter_string)
        result_pickle_path = os.path.join(result_file_path, "score_stats.pickle")
        
        thresholds_file_path = os.path.join(thresholds_folder, cutoffs_string, "X_train", method_name, hyperparameter_string)
        thresholds_pickle_path = os.path.join(thresholds_file_path, "thresholds.pickle")
        
        if not (os.path.exists(result_pickle_path) and os.path.exists(thresholds_pickle_path)):
            
            # check if all predictions have actually been calculated
            y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_train", prediction_folder, method_name, hyperparameter_string, pickle_train_file_folder)
        
            res = minimize(neg_threshold_and_score, 0, args=(y_true_filtered, y_scores_filtered, event_lengths_filtered, cutoffs), method="Nelder-Mead", options ={"disp":False})
            
            threshold = res.x[0]
            
            y_pred = threshold_scores(y_scores_filtered, threshold)
            
            score_stats  = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs, return_subscores=True, return_confmat=True)
            
            storm_score, sub_scores, TN, FP, FN, TP = score_stats
            
            os.makedirs(result_file_path, exist_ok=True)
            with open(result_pickle_path, 'wb') as handle:
                pickle.dump(score_stats, handle)
                
            os.makedirs(thresholds_file_path, exist_ok=True)
            with open(thresholds_pickle_path, 'wb') as handle:
                pickle.dump(threshold, handle)
        else:
            
            with open(result_pickle_path, 'rb') as handle:
                storm_score, sub_scores, TN, FP, FN, TP = pickle.load(handle)
            with open(thresholds_pickle_path, 'rb') as handle:
                threshold = pickle.load(handle)
            
            
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
    
    
    result_file_path = os.path.join(result_folder, cutoffs_string, "X_test", method_name, hyperparameter_string)
    result_pickle_path = os.path.join(result_file_path, "score_stats.pickle")
    
    thresholds_file_path = os.path.join(thresholds_folder, cutoffs_string, "X_test", method_name, hyperparameter_string)
    thresholds_pickle_path = os.path.join(thresholds_file_path, "thresholds.pickle")
    
    if not (os.path.exists(result_pickle_path) and os.path.exists(thresholds_pickle_path)):
        
    
        
        get_IF_scores(pickle_test_file_folder, "X_test", [best_hyperparameters])
        hyperparameter_string = str(best_hyperparameters)
        
        y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_test", prediction_folder, method_name, hyperparameter_string, pickle_test_file_folder)
        
        y_pred = threshold_scores(y_scores_filtered, best_threshold)
        
        score_stats  = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs, return_subscores=True, return_confmat=True)
        
        storm_score, sub_scores, TN, FP, FN, TP = score_stats
        
        os.makedirs(result_file_path, exist_ok=True)
        with open(result_pickle_path, 'wb') as handle:
            pickle.dump(score_stats, handle)
            
        os.makedirs(thresholds_file_path, exist_ok=True)
        with open(thresholds_pickle_path, 'wb') as handle:
            pickle.dump(best_threshold, handle)
    else:
        with open(result_pickle_path, 'rb') as handle:
            storm_score, sub_scores, TN, FP, FN, TP = pickle.load(handle)
        
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
            
            prediction_file_path = os.path.join(prediction_folder, data_name, method_name, hyperparameter_string, substation_name+".pickle")
            
            if os.path.exists(prediction_file_path):
                pass
            else:        
                print("evaluating hyperparameter setting:")
                print(hyperparameter_settings)
                #evaluate method using hyperparameter_settings
                
                model = RobustScaler(**hyperparameter_settings)
                
                y_scores = np.squeeze(model.fit_transform(X))
                
                if not os.path.exists(os.path.join(prediction_folder, data_name, method_name, hyperparameter_string)):
                    os.makedirs(os.path.join(prediction_folder, data_name, method_name, hyperparameter_string))
                
                with open(prediction_file_path, 'wb') as handle:
                    pickle.dump(y_scores, handle)


#Perform grid search
#calculate combinations from hyperparameters[method_name]
hyperparameter_grid = {"quantile_range":[(10,90), (5, 95), (2.5, 97.5), (15,85), (20,80)]} #,(25,75)

hyperparameter_list = list(ParameterGrid(hyperparameter_grid))

get_RI_scores(pickle_train_file_folder, data_name="X_train", hyperparameter_list=hyperparameter_list)

#%% evaluate Robust interval

print("Now evaluating:")
print("Robust Interval")
method_name="RI"

for cutoffs in low_and_all_cutoffs:
    cutoffs_string = str(cutoffs)
    print("Evaluating cutoffs:" + cutoffs_string)
    
    best_score = 0
    best_hyperparameters= []
    best_thresholds = 0
    
    
    print("Evaluate training data:")
    for hyperparameter_settings in hyperparameter_list:
        hyperparameter_string = str(hyperparameter_settings)
        print(hyperparameter_string)
        
        result_file_path = os.path.join(result_folder, cutoffs_string, "X_train", method_name, hyperparameter_string)
        result_pickle_path = os.path.join(result_file_path, "score_stats.pickle")
        
        thresholds_file_path = os.path.join(thresholds_folder, cutoffs_string, "X_train", method_name, hyperparameter_string)
        thresholds_pickle_path = os.path.join(thresholds_file_path, "thresholds.pickle")
    
        if not (os.path.exists(result_pickle_path) and os.path.exists(thresholds_pickle_path)):
            # check if all predictions have actually been calculated
            y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_train", prediction_folder, method_name, hyperparameter_string, pickle_train_file_folder)
        
            res = minimize(neg_double_threshold_and_score, (-1,1), args=(y_true_filtered, y_scores_filtered, event_lengths_filtered, cutoffs), method="Nelder-Mead", options ={"disp":False})
            
            thresholds = res.x
            
            y_pred = double_threshold_scores(y_scores_filtered, thresholds)
            
            score_stats  = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs, return_subscores=True, return_confmat=True)
            
            storm_score, sub_scores, TN, FP, FN, TP = score_stats
                
            os.makedirs(result_file_path, exist_ok=True)
            with open(result_pickle_path, 'wb') as handle:
                pickle.dump(score_stats, handle)
                
            os.makedirs(thresholds_file_path, exist_ok=True)
            with open(thresholds_pickle_path, 'wb') as handle:
                pickle.dump(thresholds, handle)
        else:
            
            with open(result_pickle_path, 'rb') as handle:
                storm_score, sub_scores, TN, FP, FN, TP = pickle.load(handle)
    
            with open(thresholds_pickle_path, 'rb') as handle:
                thresholds = pickle.load(handle)
    
                
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
    result_file_path = os.path.join(result_folder, cutoffs_string, "X_test", method_name, hyperparameter_string)
    
    result_pickle_path = os.path.join(result_file_path, "score_stats.pickle")
    
    thresholds_file_path = os.path.join(thresholds_folder, cutoffs_string, "X_test", method_name, hyperparameter_string)
    thresholds_pickle_path = os.path.join(thresholds_file_path, "thresholds.pickle")
    
    
    if not (os.path.exists(result_pickle_path) and os.path.exists(thresholds_pickle_path)):
        
        os.makedirs(result_file_path, exist_ok=True)
        
        get_RI_scores(pickle_test_file_folder, "X_test", [best_hyperparameters])
    
        hyperparameter_string = str(best_hyperparameters)
        
        y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_test", prediction_folder, method_name, hyperparameter_string, pickle_test_file_folder)
        
        y_pred = double_threshold_scores(y_scores_filtered, best_thresholds)
        
        score_stats  = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs, return_subscores=True, return_confmat=True)
        
        storm_score, sub_scores, TN, FP, FN, TP = score_stats
        
        os.makedirs(result_file_path, exist_ok=True)
        with open(result_pickle_path, 'wb') as handle:
            pickle.dump(score_stats, handle)
            
        os.makedirs(thresholds_file_path, exist_ok=True)
        with open(thresholds_pickle_path, 'wb') as handle:
            pickle.dump(best_thresholds, handle)
        
    else:
        with open(result_pickle_path, 'rb') as handle:
            storm_score, sub_scores, TN, FP, FN, TP = pickle.load(handle)
        
    print("STORM score on test:")
    print(storm_score)


#%% Get segments from Binseg:
    
pandas2ri.activate()

def get_BS_segments(pickle_folder, data_name, hyperparameter_list):
    for pickle_file in os.listdir(pickle_folder):
        method_name = "BS"
        substation_name = pickle_file[:-7]
        
        print("Station: ")
        print(substation_name)
        
        data = pickle.load(open(os.path.join(pickle_folder, pickle_file), 'rb'))
        X, y = pd.DataFrame(data["X"])[0], np.squeeze(data["y"].reshape(-1,1))
        
        print("evaluating hyperparameter setting:")
        for hyperparameter_settings in hyperparameter_list:
            #hyperparameter_string = re.sub(r"[^a-zA-Z0-9_ ]","",str(hyperparameter_settings))
            hyperparameter_string = str(hyperparameter_settings)
            
            intermediate_file_path = os.path.join(intermediate_folder, data_name, method_name, hyperparameter_string, substation_name+".pickle")
            
            if os.path.exists(intermediate_file_path):
                pass
            else:        

                print(hyperparameter_settings)
                #evaluate method using hyperparameter_settings
                changepoint_object = changepoint.cpt_meanvar(X, **hyperparameter_settings)
                
                changepoints = changepoint.cpts(changepoint_object).astype(int)
                
                segments = np.split(X, changepoints)
                
                if not os.path.exists(os.path.join(intermediate_folder, data_name, method_name, hyperparameter_string)):
                    os.makedirs(os.path.join(intermediate_folder, data_name, method_name, hyperparameter_string))
                
                with open(intermediate_file_path, 'wb') as handle:
                    pickle.dump(segments, handle)


#calculate combinations from hyperparameters[method_name]
#hyperparameter_grid = {"penalty":["Manual"], "pen_value":[250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000], "method":["BinSeg"], "Q":[50, 100, 200, 300, 400], "minseglen":[100, 200, 300, 400, 500]}
hyperparameter_grid = {"penalty":["Manual"], "pen_value":[50,4000], "method":["BinSeg"], "Q":[100, 400], "minseglen":[200, 500]}
#hyperparameter_grid = {"penalty":["Manual"], "pen_value":[7500, 7000], "method":["BinSeg"], "Q":[200], "minseglen":[max(2,288)]}
hyperparameter_list = list(ParameterGrid(hyperparameter_grid))

get_BS_segments(pickle_train_file_folder, data_name="X_train", hyperparameter_list=hyperparameter_list)

#%% evaluate binary segmentation based on distance of mean of segments to mean of station X
method_name="BS"
print("Now evaluating:")
print("Binary Segmentation")

for cutoffs in low_and_all_cutoffs:
    cutoffs_string = str(cutoffs)
    print("Evaluating cutoffs:" + cutoffs_string)
    
    best_score = 0
    best_hyperparameters= []
    best_thresholds = 0
    
    data_name = "X_train"
    
    
    y_true_filtered, event_lengths_filtered = get_y_true_and_lengths(pickle_train_file_folder)
    
    print("Evaluate training data:")
        
        
    for hyperparameter_settings in hyperparameter_list:
        print("Hyperparameters:")
        print(hyperparameter_settings)
        segment_features = []
        
        hyperparameter_string = str(hyperparameter_settings)
        
        result_file_path = os.path.join(result_folder, cutoffs_string, "X_train", method_name, hyperparameter_string)
        result_pickle_path = os.path.join(result_file_path, "score_stats.pickle")
        
        thresholds_file_path = os.path.join(thresholds_folder, cutoffs_string, "X_train", method_name, hyperparameter_string)
        thresholds_pickle_path = os.path.join(thresholds_file_path, "thresholds.pickle")
        
        if not (os.path.exists(result_pickle_path) and os.path.exists(thresholds_pickle_path)):
            # check if all predictions have actually been calculated
            for pickle_file in os.listdir(pickle_train_file_folder):
                substation_name = pickle_file[:-7]
                
                
                data = pickle.load(open(os.path.join(pickle_train_file_folder, pickle_file), 'rb'))
                X = pd.DataFrame(data["X"])[0]
                
                intermediate_file_path = os.path.join(intermediate_folder, data_name, method_name, hyperparameter_string, substation_name+".pickle")
                
                with open(intermediate_file_path, 'rb') as handle:
                    segments = pickle.load(handle)
                
                segment_features += [np.full(segment.shape, np.median(segment) - np.median(X)) for segment in segments]
                
            y_scores = np.concatenate(segment_features)
            
            
            print("optimizing cutoffs:")
            
            thresholds = find_BS_thresholds(y_scores, y_true_filtered, event_lengths_filtered, cutoffs)
            
            y_pred = double_threshold_scores(y_scores, thresholds)
        
            score_stats  = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs, return_subscores=True, return_confmat=True)
            
            storm_score, sub_scores, TN, FP, FN, TP = score_stats
                
            os.makedirs(result_file_path, exist_ok=True)
            with open(result_pickle_path, 'wb') as handle:
                pickle.dump(score_stats, handle)
                
            os.makedirs(thresholds_file_path, exist_ok=True)
            with open(thresholds_pickle_path, 'wb') as handle:
                pickle.dump(thresholds, handle)
        else:
            with open(result_pickle_path, 'rb') as handle:
                storm_score, sub_scores, TN, FP, FN, TP = pickle.load(handle)
    
            with open(thresholds_pickle_path, 'rb') as handle:
                thresholds = pickle.load(handle)
                
        
    
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
    result_file_path = os.path.join(result_folder, cutoffs_string, "X_test", method_name, hyperparameter_string)
    
    result_pickle_path = os.path.join(result_file_path, "score_stats.pickle")
    
    thresholds_file_path = os.path.join(thresholds_folder, cutoffs_string, "X_test", method_name, hyperparameter_string)
    thresholds_pickle_path = os.path.join(thresholds_file_path, "thresholds.pickle")
    
    data_name = "X_test"
    
    if not (os.path.exists(result_pickle_path) and os.path.exists(thresholds_pickle_path)):
        
        os.makedirs(result_file_path, exist_ok=True)
        
        get_BS_segments(pickle_test_file_folder, "X_test", [best_hyperparameters])
        
        hyperparameter_string = str(best_hyperparameters)
    
        for pickle_file in os.listdir(pickle_test_file_folder):
            substation_name = pickle_file[:-7]
            
            
            data = pickle.load(open(os.path.join(pickle_test_file_folder, pickle_file), 'rb'))
            X = pd.DataFrame(data["X"])[0]
            
            intermediate_file_path = os.path.join(intermediate_folder, data_name, method_name, hyperparameter_string, substation_name+".pickle")
            
            with open(intermediate_file_path, 'rb') as handle:
                segments = pickle.load(handle)
            
            segment_features += [np.full(segment.shape, np.median(segment) - np.median(X)) for segment in segments]
                
                
        y_scores = np.concatenate(segment_features)
            
    
        y_true_combined, event_lengths_combined = get_y_true_and_lengths(pickle_test_file_folder)
        
        
        y_true_filtered = y_true_combined[y_true_combined != 5]
        event_lengths_filtered = event_lengths_combined[y_true_combined != 5]
        y_scores_filtered = y_scores[y_true_combined != 5]
        
        #y_scores_filtered, y_true_filtered, event_lengths_filtered = get_all_station_data("X_test", prediction_folder, method_name, hyperparameter_string, pickle_test_file_folder)
        
        y_pred = double_threshold_scores(y_scores_filtered, best_thresholds)
        
        score_stats  = STORM_score(y_true_filtered, y_pred, event_lengths_filtered, cutoffs, return_subscores=True, return_confmat=True)
        
        storm_score, sub_scores, TN, FP, FN, TP = score_stats
        
        os.makedirs(result_file_path, exist_ok=True)
        with open(result_pickle_path, 'wb') as handle:
            pickle.dump(score_stats, handle)
            
        os.makedirs(thresholds_file_path, exist_ok=True)
        with open(thresholds_pickle_path, 'wb') as handle:
            pickle.dump(best_thresholds, handle)
        
    else:
        with open(result_pickle_path, 'rb') as handle:
            storm_score, sub_scores, TN, FP, FN, TP = pickle.load(handle)
        
    print("STORM score on test:")
    print(storm_score)
        
        