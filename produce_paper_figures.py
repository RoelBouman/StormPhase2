#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:58:43 2023

@author: rbouman
"""

#This script reproduces all figures from the paper.
#%% package loading
import os
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#%% Data loading

data_folder = "data"
result_folder = "results"
intermediates_folder = "intermediates"
model_folder = "saved_models"

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")
metric_folder = os.path.join(result_folder, "metrics")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

train_name = "Train"
test_name = "Test"
validation_name = "Validation"

all_dataset_names = [train_name, test_name, validation_name]

#%% Visualize/tabularize input data and preprocessing

#measurement_example.pdf


#event_length distribution.pdf
#Show as histogram
lengths = []
for split_name in all_dataset_names:
    lengths_path = os.path.join(intermediates_folder, "event_length_pickles", split_name)
    length_pickle_path = os.path.join(lengths_path, os.listdir(lengths_path)[0]) #pick first folder, as it shouldn't matter for event length distribution

    with open(length_pickle_path, 'rb') as handle:
        lengths += pickle.load(handle)

concat_lengths = np.concatenate(lengths)

unique_lengths, length_counts = np.unique(concat_lengths, return_counts=True)

normalized_length_counts = []
for unique_length, length_count in zip(unique_lengths, length_counts):#skip 0
    normalized_length_counts.append(length_count/unique_length)
#%% Visualize threshold optimization strategy


#%% Visualize/tabularize segmentation results of different methods