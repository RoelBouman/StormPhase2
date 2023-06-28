#%% Load packages
import os

import numpy as np
import pandas as pd

import pickle


from src.preprocess import preprocess_data
from src.preprocess import get_event_lengths
from src.methods import SingleThresholdStatisticalProfiling

#%% set process variables

data_folder = "data"
result_folder = "results"
intermediates_folder = "intermediates"

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

write_csv_intermediates = True

preprocessing_overwrite = False #if set to True, overwrite previous preprocessed data


#%% define local helper functions
def save_dataframe_list(dfs, station_names, folder, overwrite):
    for df, station_name in zip(dfs, station_names):
        
        file_name = os.path.join(folder, station_name)
        
        os.makedirs(folder, exist_ok = True)
        if overwrite or not os.path.exists(file_name):
            
           df.to_csv(file_name)
        
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
which_split = "Train"

#Set preprocessing settings here:
preprocessed_pickles_folder = os.path.join(intermediates_folder, "preprocessed_data_pickles", which_split)
preprocessed_csvs_folder = os.path.join(intermediates_folder, "preprocessed_data_csvs", which_split)

#TODO: preprocess_data needs rework
# - The following need to be toggles/settings:
#   - Whether to filter 'Missing' values when they are identical for N subsequent measurements
#   - Percentiles for sign correction need to be adjustable
# - The function needs only return a subset of columns (this will save substantially in memory/loading overhead)

#TODO: Add functionality to preprocess test/validation based on statistics found in train

#TODO: Name needs to change based on settings (NYI)
preprocessing_type = "basic"
preprocessed_file_name = os.path.join(preprocessed_pickles_folder, preprocessing_type + ".pickle")

if preprocessing_overwrite or not os.path.exists(preprocessed_file_name):
    print("Preprocessing X train data")
    X_train_data_preprocessed = [preprocess_data(df) for df in X_train_data]
    
    os.makedirs(preprocessed_pickles_folder, exist_ok = True)
    with open(preprocessed_file_name, 'wb') as handle:
        pickle.dump(X_train_data_preprocessed, handle)
else:
    print("Loading preprocessed X train data")
    with open(preprocessed_file_name, 'rb') as handle:
        X_train_data_preprocessed = pickle.load(handle)

if write_csv_intermediates:
    print("Writing CSV intermediates: train data")
    type_preprocessed_csvs_folder = os.path.join(preprocessed_csvs_folder, preprocessing_type)
    save_dataframe_list(X_train_data_preprocessed, X_train_files, type_preprocessed_csvs_folder, overwrite = preprocessing_overwrite)

#Preprocess Y_data AKA get the lengths of each event


event_lengths_pickles_folder = os.path.join(intermediates_folder, "event_length_pickles", which_split)
event_lengths_csvs_folder = os.path.join(intermediates_folder, "event_length_csvs", which_split)

preprocessed_file_name = os.path.join(event_lengths_pickles_folder, str(all_cutoffs) + ".pickle")
if preprocessing_overwrite or not os.path.exists(preprocessed_file_name):
    print("Preprocessing y train data")
    event_lengths = [get_event_lengths(df) for df in y_train_data]
    
    os.makedirs(event_lengths_pickles_folder, exist_ok = True)
    with open(preprocessed_file_name, 'wb') as handle:
        pickle.dump(event_lengths, handle)
else:
    print("Loading preprocessed y train data")
    with open(preprocessed_file_name, 'rb') as handle:
        event_lengths = pickle.load(handle)


#%% Detect anomalies/switch events
# Save results, make saving of scores optional, as writing this many results is fairly costly

#%% Train

#Define hyperparameter range:
    
#hyperparameter_dict = {"quantiles":[(10,90), (20,80), (25,75), (5,95)]}



method = SingleThresholdStatisticalProfiling(score_function=None)

#save best scores:
#TODO: define method for definitive selection
best_scores_path = os.path.join(score_folder, "test")
scores = method.fit_transform(X_train_data_preprocessed, y_train_data)

save_dataframe_list(scores,X_train_files, best_scores_path, overwrite=False)



#train_result_df = SP.train_result_df_
#best_model = SP.best_model_
#best_hyperparameters = SP.best_hyperparameters_
#%% Test

#%% Validation