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
import jsonpickle
import sqlite3

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from src.plot_functions import plot_S_original, plot_BU_original

sns.set()

#%% Data loading

data_folder = "data"
result_folder = "results"
intermediates_folder = "intermediates"
model_folder = "saved_models"

table_folder = "Tables"
figure_folder = "Figures"

os.makedirs(table_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")
metric_folder = os.path.join(result_folder, "metrics")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

train_name = "Train"
test_name = "Test"
validation_name = "Validation"

all_dataset_names = [train_name, test_name, validation_name]

#%% connect to database

DBFILE = "experiment_results.db"
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()

#%% Visualize/tabularize input data and preprocessing

#measurement_example.pdf
station_ID = "019"

n_xlabels = 10

train_IDs = os.listdir(os.path.join(data_folder, "Train", "X"))
test_IDs = os.listdir(os.path.join(data_folder, "Test", "X"))
validation_IDs = os.listdir(os.path.join(data_folder, "Validation", "X"))

all_station_IDs = train_IDs + test_IDs + validation_IDs

train_ID_dict = {ID.replace(".csv", ""): "Train" for ID in train_IDs}
test_ID_dict = {ID.replace(".csv", ""): "Test" for ID in test_IDs}
validation_ID_dict = {ID.replace(".csv", ""): "Validation" for ID in validation_IDs}

#fastest dict merge: https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one
station_dataset_dict = dict(train_ID_dict, **test_ID_dict)
station_dataset_dict.update(validation_ID_dict)

X_df = pd.read_csv(os.path.join(data_folder, station_dataset_dict[station_ID], "X", station_ID + ".csv"))
y_df = pd.read_csv(os.path.join(data_folder, station_dataset_dict[station_ID], "y", station_ID + ".csv"))


fig = plt.figure(figsize=(30,16)) # add DPI=300+ in case some missing points don't show up'    
plt.title("Station: " + station_ID, fontsize=60)


plot_S_original(X_df, label="S original")
plot_BU_original(X_df, label="BU original")

plt.legend(fontsize=30)

plt.yticks(fontsize=30)
plt.ylabel("S", fontsize=30)

ticks = np.linspace(0,len(X_df["S_original"])-1, n_xlabels, dtype=int)
plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=30)
plt.xlim((0, len(X_df)))

plt.tight_layout()

plt.savefig(os.path.join(figure_folder, "measurement_example.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "measurement_example.png"), format="png")
plt.show()
#event_length_distribution.pdf, event_length_stats.tex
#Show as histogram and stats table
lengths = []
for split_name in all_dataset_names:
    lengths_path = os.path.join(intermediates_folder, "event_length_pickles", split_name)
    length_pickle_path = os.path.join(lengths_path, os.listdir(lengths_path)[0]) #pick first folder, as it shouldn't matter for event length distribution

    with open(length_pickle_path, 'rb') as handle:
        lengths += pickle.load(handle)

concat_lengths = np.concatenate(lengths)

unique_lengths, length_counts = np.unique(concat_lengths, return_counts=True)


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
            
normalized_length_count_per_cutoff = {key:[value] for key, value in normalized_length_count_per_cutoff.items()}
length_count_per_cutoff = {key:[value] for key, value in length_count_per_cutoff.items()}

event_length_stats = pd.concat([pd.DataFrame(normalized_length_count_per_cutoff), pd.DataFrame(length_count_per_cutoff)])
event_length_stats.index = ["#events per cutoff", "#label $1$ per cutoff"]
event_length_stats.to_latex(buf=os.path.join(table_folder, "event_length_stats.tex"))

lengths_to_hist = []
for unique_length, normalized_length_count in zip(unique_lengths, normalized_length_counts):
    lengths_to_hist += [unique_length]*int(normalized_length_count)

plt.figure()
plt.hist(lengths_to_hist, bins=100)
plt.yscale('log')
plt.xlabel("Event Length (#data points)")
plt.ylabel("Counts (log)")
plt.savefig(os.path.join(figure_folder, "event_length_distribution.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "event_length_distribution.png"), format="png")
plt.show()

#%% Visualize threshold optimization strategy


#load model:
preprocessing_hash = "10cab9fc324db7a2fd5d8674c71edb68908b5e572ffa442d201eb0ca0aa288e1"
hyperparameter_hash = "ecf927381ef6fbb708dcf54fd846cfb856d60791b2ee998ec386c8f17661149a"

db_result = db_cursor.execute("SELECT method_hyperparameters FROM experiment_results WHERE preprocessing_hash='{}' AND hyperparameter_hash='{}'".format(preprocessing_hash, hyperparameter_hash)).fetchone()[0]

hyperparameters = jsonpickle.loads(db_result)

from src.evaluation import f_beta
from src.methods import SingleThresholdStatisticalProfiling
beta = 1.5
def score_function(precision, recall):
    return f_beta(precision, recall, beta)


model = SingleThresholdStatisticalProfiling(model_folder, preprocessing_hash, **hyperparameters, score_function=score_function)


plt.figure()

category_labels = ["1-24", "25-288", "289-4032", "4033 and longer"]
colors = sns.color_palette()[:3] + [sns.color_palette()[4]]
thresholds = model.thresholds
for i, column in enumerate(model.scores.columns):
    scores = model.scores[column]
    
    plt.plot(thresholds, scores, label=category_labels[i], linestyle=":", color=colors[i])
    
plt.plot(thresholds, np.mean(model.scores, axis=1), label="average", color=sns.color_palette()[3])

plt.axvline(x = model.optimal_threshold, color = sns.color_palette()[3], linestyle="--", label = 'optimal threshold')

plt.legend()
plt.xlabel("Threshold")
plt.ylabel("F1.5")
plt.xlim((0, 4)) #hardcoded
plt.savefig(os.path.join(figure_folder, "threshold_optimization.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "threshold_optimization.png"), format="png")
plt.show()

#%% summarize results on validation set for each method
#show for each method:
    # validation average F1.5
    # validation F1.5 for each cutoff/category
    # validation average ROC/AUC (optional?)
    # best preprocessing hyperparameters (optional?)
    # best hyperparameters (optional?)


validation_results = db_cursor.execute("SELECT method, method_hyperparameters, MAX(metric), preprocessing_hash, hyperparameter_hash FROM experiment_results WHERE which_split='Validation' GROUP BY method").fetchall()

validation_results_df = pd.DataFrame(validation_results).iloc[:,:3]
validation_results_df.columns = ["Method", "Hyperparameters", "Average F1.5"]
validation_results_df.set_index("Method", inplace=True)

for i in range(validation_results_df.shape[0]):
    #Treat ensembles differently:
    if validation_results_df.index[i] in ["NaiveEnsemble", "StackEnsemble", "SequentialEnsemble"]:
        method_dict = jsonpickle.loads(validation_results_df["Hyperparameters"].iloc[i])
        hyperparameter_strings = []
        for method, hyperparameters in zip(method_dict["method_classes"], method_dict["method_hyperparameter_dict_list"]):
            method_name = method("temp","temp").method_name
            hyperparameter_strings.append(method_name + ":" + str(hyperparameters))
        validation_results_df["Hyperparameters"].iloc[i] = "\n".join(hyperparameter_strings)
    else:
        hyperparameters = jsonpickle.loads(validation_results_df["Hyperparameters"].iloc[i])
        hyperparameters.pop("used_cutoffs", None)
        validation_results_df["Hyperparameters"].iloc[i] = str(hyperparameters)
        
validation_results_df.to_latex(buf=os.path.join(table_folder, "validation_results_df.tex"))
        
#%% Visualize/tabularize segmentation results of different methods

