#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:23:00 2024

@author: rbouman
"""
import os
from sklearn.model_selection import ParameterGrid
from hashlib import sha256
import numpy as np
import pandas as pd

from src.io_functions import load_batch
from src.preprocess import preprocess_per_batch_and_write

raw_data_folder = "raw_data"
dataset = "route_data" #alternatively: route_data
intermediates_folder = os.path.join(raw_data_folder, dataset+"_preprocessed")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

#%%

def split_series(series):
    # Sort the series in descending order
    sorted_series = series.sort_values(ascending=False)

    # Initialize three empty dictionaries to store the parts
    part_1, part_2, part_3 = {}, {}, {}

    # Iterate through the sorted series and allocate items to each part in a round-robin fashion
    for i, count in enumerate(sorted_series):
        if i % 3 == 0:
            part_1[sorted_series.index[i]] = count
        elif i % 3 == 1:
            part_2[sorted_series.index[i]] = count
        else:
            part_3[sorted_series.index[i]] = count

    # Create Series for each part
    series_part_1 = pd.Series(part_1, name='Count')
    series_part_2 = pd.Series(part_2, name='Count')
    series_part_3 = pd.Series(part_3, name='Count')

    return series_part_1.index, series_part_2.index, series_part_3.index

# %%

station_exclude_list = [25, 106, 130, 190] #106 has already been deleted beforehand due to being an invalid file

station_exclude_list_filenames = [str(station)+".csv" for station in station_exclude_list]
#%% calculate event lengths
X_dfs, y_dfs, file_names = load_batch(raw_data_folder, dataset)

#filter exclude list:
filtered_data = [(X_df, y_df, file_name) for X_df, y_df, file_name in zip(X_dfs, y_dfs, file_names) if file_name not in station_exclude_list_filenames]

filtered_X_dfs, filtered_y_dfs, filtered_file_names = zip(*filtered_data)


all_preprocessing_hyperparameters = {'subsequent_nr': [5], 'lin_fit_quantiles': [(10, 90)]}

preprocessing_hyperparameters = list(ParameterGrid(all_preprocessing_hyperparameters))[0]

preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()

_, _, _, event_lengths = preprocess_per_batch_and_write(filtered_X_dfs, filtered_y_dfs, intermediates_folder, dataset, preprocessing_overwrite=False, write_csv_intermediates=False, file_names=filtered_file_names, all_cutoffs=all_cutoffs, hyperparameters=preprocessing_hyperparameters, hyperparameter_hash=preprocessing_hash, remove_missing=True)

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
    
# #%%
# smallest_category = total_normalized_length_count_per_cutoff.index[total_normalized_length_count_per_cutoff.argmin()]

# stations_in_category_with_events = normalized_lengths_df[smallest_category].iloc[normalized_lengths_df[smallest_category].values.nonzero()]

# (train_stations, val_stations, test_stations) = split_series(stations_in_category_with_events)
# all_train_stations  += list(train_stations)
# all_val_stations += list(val_stations)
# all_test_stations += list(test_stations)
# all_considered_stations = all_train_stations + all_val_stations + all_test_stations

# remaining_stations = [file_name for file_name in filtered_file_names if file_name not in all_considered_stations]
# remaining_normalized_lengths_df = normalized_lengths_df.loc[remaining_stations]

# previous_categories = [smallest_category]

#%% divide subsequent category:
    
sorted_categories = total_normalized_length_count_per_cutoff.sort_values().index(sorted_categories)

for smallest_category in sorted_categories:
    
not_considered_categories = [category for category in total_normalized_length_count_per_cutoff.index if category not in previous_categories]

smallest_category = total_normalized_length_count_per_cutoff.index[total_normalized_length_count_per_cutoff.loc[not_considered_categories].argmin()]

stations_in_category_with_events = remaining_normalized_lengths_df[smallest_category].iloc[remaining_normalized_lengths_df[smallest_category].values.nonzero()]

(train_stations, val_stations, test_stations) = split_series(stations_in_category_with_events)
all_train_stations  += list(train_stations)
all_val_stations += list(val_stations)
all_test_stations += list(test_stations)
all_considered_stations = all_train_stations + all_val_stations + all_test_stations

remaining_stations = [file_name for file_name in filtered_file_names if file_name not in all_considered_stations]
remaining_normalized_lengths_df = normalized_lengths_df.loc[remaining_stations]

previous_categories += [smallest_category]