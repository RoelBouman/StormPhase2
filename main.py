#%% Load packages
import os

import numpy as np
import pandas as pd

import pickle


from src.preprocess import preprocess_data

#%% set process variables

data_folder = "data"
result_folder = "results"
intermediates_folder = "intermediates"


all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

write_csv_intermediates = True

preprocessing_overwrite = False #if set to True, overwrite previous preprocessed data

#%% load data
# Do not load data if preprocessed data is available already
X_train_path = os.path.join(data_folder, "Train", "X")
y_train_path = os.path.join(data_folder, "Train", "y")

X_train_files = sorted(os.listdir(X_train_path))
y_train_files = sorted(os.listdir(y_train_path))

if not X_train_files == y_train_files:
    raise RuntimeError("Not all training files are present in both the X and y folders.")


X_train_data, y_train_data = [], []
for file in X_train_files:
    X_train_data.append(pd.read_csv(os.path.join(X_train_path, file)))
    y_train_data.append(pd.read_csv(os.path.join(y_train_path, file)))
    
#%% Preprocess data
# Peprocess entire batch
# Save preprocessed data for later recalculations

#Set preprocessing settings here:
preprocessed_pickles_folder = os.path.join(intermediates_folder, "preprocessed_data_pickles", "Train")
preprocessed_csvs_folder = os.path.join(intermediates_folder, "preprocessed_data_csvs", "Train")

#TODO: preprocess_data needs rework
# - The following need to be toggles/settings:
#   - Whether to filter 'Missing' values when they are identical for N subsequent measurements
#   - Percentiles for sign correction need to be adjustable
# - The function needs only return a subset of columns (this will save substantially in memory/loading overhead)

# Name needs to change based on settings (NYI)
preprocessing_type = "basic"
preprocessed_file_name = os.path.join(preprocessed_pickles_folder, preprocessing_type + ".pickle")

if preprocessing_overwrite or not os.path.exists(preprocessed_file_name):
    print("Preprocessing train data")
    X_train_data_preprocessed = [preprocess_data(df) for df in X_train_data]
    
    os.makedirs(preprocessed_pickles_folder, exist_ok = True)
    with open(preprocessed_file_name, 'wb') as handle:
        pickle.dump(X_train_data_preprocessed, handle)
else:
    print("Loading preprocessed train data")
    with open(preprocessed_file_name, 'rb') as handle:
        X_train_data_preprocessed = pickle.load(handle)

if write_csv_intermediates:
    print("Writing CSV intermediates: train data")
    for X_train_data_preprocessed_df, file in zip(X_train_data_preprocessed, X_train_files):
        
        preprocessed_file_name = os.path.join(preprocessed_csvs_folder, preprocessing_type, file)
        
        os.makedirs(os.path.join(preprocessed_csvs_folder, preprocessing_type), exist_ok = True)
        if preprocessing_overwrite or not os.path.exists(preprocessed_file_name):
            
           X_train_data_preprocessed_df.to_csv(preprocessed_file_name)



#%% Detect anomalies/switch events
# Save results, make saving of scores optional, as writing this many results is fairly costly

#%% Train

#Define hyperparameter range:
    
hyperparameter_dict = None

method = SP()

SP.train(hyperparameter_dict=hyperparameter_dict)

train_result_df = SP.train_result_df_
best_model = SP.best_model_
best_hyperparameters = SP.best_hyperparameters_
#%% Test

#%% Validation