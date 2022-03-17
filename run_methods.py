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
        
        result_file_path = os.path.join(result_folder, method_name, hyperparameter_string, substation_name+".pickle")
        
        if os.path.exists(result_file_path):
            pass
        else:        
            print("evaluating hyperparameter setting:")
            print(hyperparameter_settings)
            #evaluate method using hyperparameter_settings
            
            model = IsolationForest(**hyperparameter_settings)
            model.fit(X)
            y_scores = model.decision_function(X)
            
            if not os.path.exists(os.path.join(result_folder, method_name, hyperparameter_string)):
                os.makedirs(os.path.join(result_folder, method_name, hyperparameter_string))
            
            with open(result_file_path, 'wb') as handle:
                pickle.dump(y_scores, handle)
            

#%% Isolation Forest score evaluation
print("Now evaluating:")
print("Isolation Forest")

method_name = "IF"

for hyperparameter_settings in hyperparameter_list:
    hyperparameter_string = str(hyperparameter_settings)