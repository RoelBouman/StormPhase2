#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:23:00 2024

@author: rbouman
"""
import os
import shutil
from sklearn.model_selection import ParameterGrid
from hashlib import sha256
import numpy as np
import pandas as pd

from src.io_functions import load_batch
from src.preprocess import preprocess_per_batch_and_write

raw_data_folder = "raw_data"
processed_data_folder = "data"


dataset = "route_data" #alternatively: route_data
intermediates_folder = os.path.join(raw_data_folder, dataset+"_preprocessed")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

#%%

def split_series(series, counts=np.zeros((3,))):
    # Sort the series in descending order
    sorted_series = series.sort_values(ascending=False)

    # Initialize three empty dictionaries to store the parts
    parts = [{}, {}, {}]

    # Iterate through the sorted series and allocate items to each part in a round-robin fashion
    for i, count in enumerate(sorted_series):
        
        min_counts_part = np.argmin(counts)
        
        parts[min_counts_part][sorted_series.index[i]] = count
        counts[min_counts_part] += count

    # Create Series for each part
    series_part_1 = pd.Series(parts[0], name='Count')
    series_part_2 = pd.Series(parts[1], name='Count')
    series_part_3 = pd.Series(parts[2], name='Count')

    return series_part_1.index, series_part_2.index, series_part_3.index

# %%
if dataset == "route_data":
    station_exclude_list = [25, 106, 130, 190] #106 has already been deleted beforehand due to being an invalid file
elif dataset == "OS_data":
    station_exclude_list = []
else:
    station_exclude_list = []
    
station_exclude_list_filenames = [str(station)+".csv" for station in station_exclude_list]
#%% calculate event lengths
X_dfs, y_dfs, file_names = load_batch(raw_data_folder, dataset)

#filter exclude list:
filtered_data = [(X_df, y_df, file_name) for X_df, y_df, file_name in zip(X_dfs, y_dfs, file_names) if file_name not in station_exclude_list_filenames]

filtered_X_dfs, filtered_y_dfs, filtered_file_names = zip(*filtered_data)

filtered_X_dfs, filtered_y_dfs, filtered_file_names = list(filtered_X_dfs), list(filtered_y_dfs), list(filtered_file_names)

all_preprocessing_hyperparameters = {'subsequent_nr': [5], 'lin_fit_quantiles': [(10, 90)]}

preprocessing_hyperparameters = list(ParameterGrid(all_preprocessing_hyperparameters))[0]

preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()

_, _, _, event_lengths = preprocess_per_batch_and_write(filtered_X_dfs, filtered_y_dfs, intermediates_folder, dataset, preprocessing_overwrite=False, write_csv_intermediates=False, file_names=filtered_file_names, all_cutoffs=all_cutoffs, hyperparameters=preprocessing_hyperparameters, hyperparameter_hash=preprocessing_hash, remove_missing=True)

#%% 
normalized_length_count_per_cutoff_dict = {}
length_count_per_cutoff_dict = {}

for event_length, file_name in zip(event_lengths, filtered_file_names):
        
    unique_lengths, length_counts = np.unique(event_length, return_counts=True)
    
    unique_lengths = unique_lengths[1:]
    length_counts = length_counts[1:]
    
    normalized_length_counts = []
    for unique_length, length_count in zip(unique_lengths, length_counts):#skip 0
        normalized_length_counts.append(length_count/unique_length)
        
    normalized_length_count_per_cutoff = {str(cutoff):0 for cutoff in all_cutoffs}
    length_count_per_cutoff = {str(cutoff):0 for cutoff in all_cutoffs}
    
    for unique_length, normalized_length_count, length_count in zip(unique_lengths,normalized_length_counts, length_counts):
        for cutoff in all_cutoffs:
            if unique_length >= cutoff[0] and unique_length < cutoff[1]: 
                normalized_length_count_per_cutoff[str(cutoff)] += normalized_length_count
                length_count_per_cutoff[str(cutoff)] += length_count
                
    normalized_length_count_per_cutoff_dict[file_name] = {key:[value][0] for key, value in normalized_length_count_per_cutoff.items()}
    length_count_per_cutoff_dict[file_name] = {key:[value][0] for key, value in length_count_per_cutoff.items()}
    
normalized_lengths_df = pd.DataFrame(normalized_length_count_per_cutoff_dict).T
lengths_df = pd.DataFrame(length_count_per_cutoff_dict).T

total_normalized_length_count_per_cutoff = normalized_lengths_df.sum()
total_length_count_per_cutoff = lengths_df.sum()

#%% start dividing datasets by category with lowest normalized length count:
all_train_stations, all_val_stations, all_test_stations, all_considered_stations = [], [], [], []
previous_categories = []
remaining_normalized_lengths_df = normalized_lengths_df
    

#%% divide over each category, starting with the smallest:
    
sorted_categories = total_normalized_length_count_per_cutoff.sort_values().index

for smallest_category in sorted_categories:
    
    stations_in_category_with_events = remaining_normalized_lengths_df[smallest_category].iloc[remaining_normalized_lengths_df[smallest_category].values.nonzero()]
    
    #get current counts:
    counts = np.zeros((3,))
    for i, stations in enumerate((all_train_stations, all_val_stations, all_test_stations)):
        counts[i] = (normalized_lengths_df.loc[stations].sum())[smallest_category]
        
    print(counts)
    #print(len(stations_in_category_with_events))
    (train_stations, val_stations, test_stations) = split_series(stations_in_category_with_events, counts)
    all_train_stations  += list(train_stations)
    all_val_stations += list(val_stations)
    all_test_stations += list(test_stations)
    all_considered_stations = all_train_stations + all_val_stations + all_test_stations
    
    remaining_stations = [file_name for file_name in filtered_file_names if file_name not in all_considered_stations]
    remaining_normalized_lengths_df = normalized_lengths_df.loc[remaining_stations]
    
    previous_categories += [smallest_category]
    
    

#%% Divide remaining stations (with no events)

stations_in_category_with_events_index = remaining_normalized_lengths_df.index

stations_in_category_with_events = pd.Series(np.ones(len(stations_in_category_with_events_index)), index=stations_in_category_with_events_index)

counts = np.array([len(all_train_stations), len(all_val_stations), len(all_test_stations)])
(train_stations, val_stations, test_stations) = split_series(stations_in_category_with_events, counts)
all_train_stations  += list(train_stations)
all_val_stations += list(val_stations)
all_test_stations += list(test_stations)
all_considered_stations = all_train_stations + all_val_stations + all_test_stations

#%% sanity check to see how well division was done:
print("Events in Train, Val, Test:")
for stations in (all_train_stations, all_val_stations, all_test_stations):
    print(normalized_lengths_df.loc[stations].sum())
    print(len(stations))
    
#%% Save data to folder based on calculated split:
stations_per_split = {"Train":all_train_stations, "Validation":all_val_stations, "Test": all_test_stations}

for split, station_names in stations_per_split.items():
    for station_name in station_names:
        original_X_file = os.path.join(raw_data_folder, dataset, "X", station_name)
        new_X_file = os.path.join(processed_data_folder, dataset, split, "X", station_name)
        
        original_y_file = os.path.join(raw_data_folder, dataset, "y", station_name)
        new_y_file = os.path.join(processed_data_folder, dataset, split, "y", station_name)
        
        os.makedirs(os.path.join(processed_data_folder, dataset, split, "X"), exist_ok=True)
        os.makedirs(os.path.join(processed_data_folder, dataset, split, "y"), exist_ok=True)
        
        shutil.copy(original_X_file, new_X_file)
        shutil.copy(original_y_file, new_y_file)
    