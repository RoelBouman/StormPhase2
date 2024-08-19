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
import matplotlib as mpl
import seaborn as sns



from src.plot_functions import plot_S_original, plot_BU_original
from src.io_functions import load_table, load_metric, load_minmax_stats

from src.methods import SingleThresholdStatisticalProcessControl
from src.methods import DoubleThresholdStatisticalProcessControl
from src.methods import SingleThresholdIsolationForest

from src.methods import SingleThresholdBinarySegmentation
from src.methods import DoubleThresholdBinarySegmentation

from src.methods import StackEnsemble
from src.methods import NaiveStackEnsemble
from src.methods import SequentialEnsemble


sns.set()



#%% Data loading

dataset = "OS_data" #alternatively: route_data
data_folder = os.path.join("data", dataset)
result_folder = os.path.join("results", dataset)
intermediates_folder = os.path.join("intermediates", dataset)
model_folder = os.path.join("saved_models", dataset)

table_folder = os.path.join("Tables", dataset)
figure_folder = os.path.join("Figures", dataset)

os.makedirs(table_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")
metric_folder = os.path.join(result_folder, "metrics")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

train_name = "Train"
test_name = "Test"
validation_name = "Validation"

all_dataset_names = [train_name, validation_name, test_name]

#%% connect to database

DBFILE = dataset+"_experiment_results.db" #TODO: remove _copy part
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()

#%% replace cutoff by category:
cutoff_replacement_dict = {"(0, 24)":"15m-6h", "(24, 288)":"6h-3d","(288, 4032)":"3d-42d", "(4032, inf)":"42d and longer"}

#%% Visualize/tabularize input data and preprocessing

#%% measurement_example.pdf (normal)
station_ID = "005"

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
#plt.title("Station: " + station_ID, fontsize=60)


plot_S_original(pd.DataFrame(X_df["S_original"]/1000), label="S original")
plot_BU_original(pd.DataFrame(X_df["BU_original"]/1000), label="BU original")

X_max = np.max(X_df["S_original"])/1000
X_min = np.min(X_df["S_original"])/1000

T_0 = 0
T_end = len(X_df["M_TIMESTAMP"])

plt.axhline(y=X_max, color='black', linestyle='dashed', label="Maximum/minimum measured load")
plt.axhline(y=X_min, color='black', linestyle='dashed')

#Plot unused capacity and redundancy as rectangles:
max_capacity = 90000/1000
redundant_capacity=30000/1000
    
max_unused_capacity = max_capacity-redundant_capacity-X_max
min_unused_capacity = -max_capacity+redundant_capacity-X_min

opacity = 0.3

ax = plt.gca()
#Plot unused capacity patches
max_unused_capacity_patch = mpl.patches.Rectangle(xy=(T_0, X_max), height=max_unused_capacity, width = T_end)
min_unused_capacity_patch = mpl.patches.Rectangle(xy=(T_0, X_min), height=min_unused_capacity, width = T_end)

pc = mpl.collections.PatchCollection([max_unused_capacity_patch, min_unused_capacity_patch], facecolor="g", alpha=opacity)
ax.add_collection(pc)
unused_capacity_handle = mpl.patches.Patch(color='g', alpha=opacity, label='Unused capacity')

# #Plot redundant capacity patches
max_redundant_capacity_patch = mpl.patches.Rectangle(xy=(T_0, X_max+max_unused_capacity), height=redundant_capacity, width = T_end)
min_redundant_capacity_patch = mpl.patches.Rectangle(xy=(T_0, X_min+min_unused_capacity), height=-redundant_capacity, width = T_end)

pc = mpl.collections.PatchCollection([max_redundant_capacity_patch, min_redundant_capacity_patch], facecolor="b", alpha=opacity)
ax.add_collection(pc)
redundant_capacity_handle = mpl.patches.Patch(color='b', alpha=opacity, label='Redundant capacity')

plt.axhline(y=max_capacity, color='black', linestyle='dotted', linewidth=4, label="Load limit")
plt.axhline(y=-max_capacity, color='black', linestyle='dotted', linewidth=4)

existing_handles, _ = ax.get_legend_handles_labels()
# plt.legend(handles=existing_handles+[unused_capacity_handle], fontsize=30)
plt.legend(handles=existing_handles+[unused_capacity_handle, redundant_capacity_handle], fontsize=40)
# plt.legend(fontsize=30)


plt.yticks(fontsize=40)
plt.ylabel("load (MW)", fontsize=40)

ticks = np.linspace(0,len(X_df["S_original"])-1, n_xlabels, dtype=int)
plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=40)
plt.xlim((0, len(X_df)))

plt.ylim((-max_capacity-5000/1000, max_capacity+5000/1000))

plt.tight_layout()

plt.savefig(os.path.join(figure_folder, "measurement_example.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "measurement_example.png"), format="png")
plt.show()

#%% switch event illustration plot

no_switch_data = np.repeat(5,1000)

negative_switch_data = np.concatenate([np.repeat(5,500), np.repeat(1,500)])
positive_switch_data = np.concatenate([np.repeat(5,500), np.repeat(9,500)])


plt.figure(figsize=(6,4))
plt.plot(no_switch_data,  "black")
plt.xlabel("time", fontsize=30)
plt.ylabel("S", fontsize=30)
plt.xlim(0,1000)
plt.ylim(0,10)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(False)
plt.tight_layout()

plt.savefig(os.path.join(figure_folder, "no_switch_event_example.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "no_switch_event_example.png"), format="png")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(negative_switch_data,  "black")
plt.xlabel("time", fontsize=30)
plt.ylabel("S", fontsize=30)
plt.xlim(0,1000)
plt.ylim(0,10)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(False)
plt.tight_layout()

plt.savefig(os.path.join(figure_folder, "negative_switch_event_example.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "negative_switch_event_example.png"), format="png")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(positive_switch_data,  "black")
plt.xlabel("time", fontsize=30)
plt.ylabel("S", fontsize=30)
plt.xlim(0,1000)
plt.ylim(0,10)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(False)
plt.tight_layout()

plt.savefig(os.path.join(figure_folder, "positive_switch_event_example.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "positive_switch_event_example.png"), format="png")
plt.show()



#%% measurement_example.pdf (normal)
station_ID = "018"

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
#plt.title("Station: " + station_ID, fontsize=60)


plot_S_original(pd.DataFrame(X_df["S_original"]/1000), label="S original")
plot_BU_original(pd.DataFrame(X_df["BU_original"]/1000), label="BU original")

X_max = np.max(X_df["S_original"])/1000
X_min = np.min(X_df["S_original"])/1000

X_max_true = np.max(X_df["S_original"].loc[y_df["label"]==0])/1000
X_min_true = np.min(X_df["S_original"].loc[y_df["label"]==0])/1000

T_0 = 0
T_end = len(X_df["M_TIMESTAMP"])

plt.axhline(y=X_max, color='black', linestyle='dashed', label="Maximum/minimum measured load")
plt.axhline(y=X_min, color='black', linestyle='dashed')




plt.axhline(y=X_max_true, color='red', linestyle='dashed', label="True maximum/minimum load")
plt.axhline(y=X_min_true, color='red', linestyle='dashed')

#Plot unused capacity and redundancy as rectangles:
max_capacity = 120000/1000
redundant_capacity = 20/1000
    
#max_unused_capacity = max_capacity-redundant_capacity-X_max
min_unused_capacity = -max_capacity+redundant_capacity-X_min

opacity = 0.3

ax = plt.gca()
#Plot unused capacity patches
incorrect_max_capacity = mpl.patches.Rectangle(xy=(T_0, X_max_true), height=X_max-X_max_true, width = T_end)
incorrect_min_capacity = mpl.patches.Rectangle(xy=(T_0, X_min), height=X_min_true-X_min, width = T_end)
#min_unused_capacity_patch = mpl.patches.Rectangle(xy=(T_0, X_min), height=min_unused_capacity, width = T_end)

pc = mpl.collections.PatchCollection([incorrect_max_capacity, incorrect_min_capacity], facecolor="r", alpha=opacity)

ax.add_collection(pc)
incorrect_capacity_handle = mpl.patches.Patch(color='r', alpha=opacity, label='Incorrectly estimated capacity')

existing_handles, _ = ax.get_legend_handles_labels()
# plt.legend(handles=existing_handles+[unused_capacity_handle], fontsize=30)
plt.legend(handles=existing_handles+[incorrect_capacity_handle], fontsize=40)

#plt.legend(fontsize=30)


plt.yticks(fontsize=40)
plt.ylabel("load (MW)", fontsize=40)

ticks = np.linspace(0,len(X_df["S_original"])-1, n_xlabels, dtype=int)
plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=40)
plt.xlim((0, len(X_df)))

#plt.ylim((-max_capacity-5000/1000, max_capacity+5000/1000))

plt.tight_layout()

plt.savefig(os.path.join(figure_folder, "measurement_example_anomaly.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "measurement_example_anomaly.png"), format="png")
plt.show()

#%%event_length_distribution.pdf, event_length_stats.tex
#Show as histogram and stats table
lengths = {}
for split_name in all_dataset_names:
    lengths_path = os.path.join(intermediates_folder, "event_length_pickles", split_name)
    length_pickle_path = os.path.join(lengths_path, os.listdir(lengths_path)[0]) #pick first folder, as it shouldn't matter for event length distribution

    with open(length_pickle_path, 'rb') as handle:
        lengths[split_name] = pickle.load(handle)

lengths["All"] = [df for dfs_list in lengths.values() for df in dfs_list]

normalized_length_count_per_cutoff_list = []
length_count_per_cutoff_list = []

for split in all_dataset_names+["All"]:
    concat_lengths = np.concatenate(lengths[split])
    
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
                
    normalized_length_count_per_cutoff_list.append({key:[value][0] for key, value in normalized_length_count_per_cutoff.items()})
    length_count_per_cutoff_list.append({key:[value][0] for key, value in length_count_per_cutoff.items()})

event_length_stats = pd.concat([pd.DataFrame(normalized_length_count_per_cutoff_list), pd.DataFrame(length_count_per_cutoff_list)]).astype(int)
event_length_stats.index = pd.MultiIndex.from_product([["Event count", "Label $1$ count"], list(lengths.keys())], names=["", "Dataset"])
    
    
event_length_stats.rename(columns=cutoff_replacement_dict, inplace=True)
event_length_stats.to_latex(buf=os.path.join(table_folder, "event_length_stats.tex"), escape=False, multirow=True)

lengths_to_hist = []
for unique_length, normalized_length_count in zip(unique_lengths, normalized_length_counts):
    lengths_to_hist += [unique_length]*int(normalized_length_count)

plt.figure()
plt.hist(lengths_to_hist, bins=100)
plt.yscale('log')
plt.xlabel("Event Length (#data points)")
plt.ylabel("Counts")
plt.savefig(os.path.join(figure_folder, "event_length_distribution.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "event_length_distribution.png"), format="png")
plt.show()

#%% Visualize threshold optimization strategy


#load model:
preprocessing_hash = "ef19085e70a2b043dd00e10361154f3ec54122c056f0a5236099c900ff889eff"
hyperparameter_hash = "863c7a1a49f110ada1d11bf21549b9f60f53c72042a80a36a0969583a18d42e1"

db_result = db_cursor.execute("SELECT method_hyperparameters FROM experiment_results WHERE preprocessing_hash='{}' AND hyperparameter_hash='{}'".format(preprocessing_hash, hyperparameter_hash)).fetchone()[0]

hyperparameters = jsonpickle.loads(db_result)


from src.evaluation import f_beta

beta = 1.5
def score_function(precision, recall):
    return f_beta(precision, recall, beta)


model = SingleThresholdStatisticalProcessControl(model_folder, preprocessing_hash, **hyperparameters, score_function=score_function)
model.calculate_and_set_thresholds(all_cutoffs)

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
plt.xlabel(r"Threshold ($\theta$)")
plt.ylabel(r"F1.5")
plt.xlim((0, 4)) #hardcoded
plt.savefig(os.path.join(figure_folder, "threshold_optimization.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "threshold_optimization.png"), format="png")
plt.show()



#%% Bar plot of bootstrapping results on test:

which_split = "Test"
    
methods = { "SingleThresholdIF":SingleThresholdIsolationForest,
            "SingleThresholdBS":SingleThresholdBinarySegmentation, 
            "SingleThresholdSPC":SingleThresholdStatisticalProcessControl,
            
            "DoubleThresholdBS":DoubleThresholdBinarySegmentation, 
            "DoubleThresholdSPC":DoubleThresholdStatisticalProcessControl, 
            
            "Naive-SingleThresholdBS+SingleThresholdSPC":NaiveStackEnsemble, 
            "Naive-DoubleThresholdBS+DoubleThresholdSPC":NaiveStackEnsemble,
            "Naive-SingleThresholdBS+DoubleThresholdSPC":NaiveStackEnsemble,
            "Naive-DoubleThresholdBS+SingleThresholdSPC":NaiveStackEnsemble,
            
            "Naive-SingleThresholdBS+SingleThresholdIF":NaiveStackEnsemble,
            "Naive-DoubleThresholdBS+SingleThresholdIF":NaiveStackEnsemble,
                
            "SingleThresholdBS+SingleThresholdSPC":StackEnsemble, 
            "DoubleThresholdBS+DoubleThresholdSPC":StackEnsemble, 
            "SingleThresholdBS+DoubleThresholdSPC":StackEnsemble,
            "DoubleThresholdBS+SingleThresholdSPC":StackEnsemble,
            
            "SingleThresholdBS+SingleThresholdIF":StackEnsemble,
            "DoubleThresholdBS+SingleThresholdIF":StackEnsemble,
            
            "Sequential-SingleThresholdBS+SingleThresholdSPC":SequentialEnsemble, 
            "Sequential-DoubleThresholdBS+DoubleThresholdSPC":SequentialEnsemble,
            "Sequential-SingleThresholdBS+DoubleThresholdSPC":SequentialEnsemble,
            "Sequential-DoubleThresholdBS+SingleThresholdSPC":SequentialEnsemble,
            
            "Sequential-SingleThresholdBS+SingleThresholdIF":SequentialEnsemble, 
            "Sequential-DoubleThresholdBS+SingleThresholdIF":SequentialEnsemble,
            }

name_abbreviations = { 
    "SingleThresholdIF": "ST-IF",
    "SingleThresholdBS": "ST-BS",
    "SingleThresholdSPC": "ST-SPC",
    "DoubleThresholdBS": "DT-BS",
    "DoubleThresholdSPC": "DT-SPC",
    "Naive-SingleThresholdBS+SingleThresholdSPC": "Naive ST-BS+ST-SPC",
    "Naive-DoubleThresholdBS+DoubleThresholdSPC": "Naive DT-BS+DT-SPC",
    "Naive-SingleThresholdBS+DoubleThresholdSPC": "Naive ST-BS+DT-SPC",
    "Naive-DoubleThresholdBS+SingleThresholdSPC": "Naive DT-BS+ST-SPC",
    "Naive-SingleThresholdBS+SingleThresholdIF": "Naive ST-BS+ST-IF",
    "Naive-DoubleThresholdBS+SingleThresholdIF": "Naive DT-BS+ST-IF",
    
    "SingleThresholdBS+SingleThresholdSPC": "DOC ST-BS+ST-SPC",
    "DoubleThresholdBS+DoubleThresholdSPC": "DOC DT-BS+DT-SPC",
    "SingleThresholdBS+DoubleThresholdSPC": "DOC ST-BS+DT-SPC",
    "DoubleThresholdBS+SingleThresholdSPC": "DOC DT-BS+ST-SPC",
    "SingleThresholdBS+SingleThresholdIF": "DOC ST-BS+ST-IF",
    "DoubleThresholdBS+SingleThresholdIF": "DOC DT-BS+ST-IF",
    
    "Sequential-SingleThresholdBS+SingleThresholdSPC": "Seq ST-BS+ST-SPC",
    "Sequential-DoubleThresholdBS+DoubleThresholdSPC": "Seq DT-BS+DT-SPC",
    "Sequential-SingleThresholdBS+DoubleThresholdSPC": "Seq ST-BS+DT-SPC",
    "Sequential-DoubleThresholdBS+SingleThresholdSPC": "Seq DT-BS+ST-SPC",
    "Sequential-SingleThresholdBS+SingleThresholdIF":"Seq ST-BS+ST-IF", 
    "Sequential-DoubleThresholdBS+SingleThresholdIF":"Seq DT-BS+ST-IF",
}

method_groups = { 
    "SingleThresholdIF": "IF",
    "SingleThresholdBS": "BS",
    "SingleThresholdSPC": "SPC",
    "DoubleThresholdBS": "BS",
    "DoubleThresholdSPC": "SPC",
    
    "Naive-SingleThresholdBS+SingleThresholdSPC": "Naive BS+SPC",
    "Naive-DoubleThresholdBS+DoubleThresholdSPC": "Naive BS+SPC",
    "Naive-SingleThresholdBS+DoubleThresholdSPC": "Naive BS+SPC",
    "Naive-DoubleThresholdBS+SingleThresholdSPC": "Naive BS+SPC",
    "Naive-SingleThresholdBS+SingleThresholdIF": "Naive BS+IF",
    "Naive-DoubleThresholdBS+SingleThresholdIF": "Naive BS+IF",
    
    "SingleThresholdBS+SingleThresholdSPC": "DOC BS+SPC",
    "DoubleThresholdBS+DoubleThresholdSPC": "DOC BS+SPC",
    "SingleThresholdBS+DoubleThresholdSPC": "DOC BS+SPC",
    "DoubleThresholdBS+SingleThresholdSPC": "DOC BS+SPC",
    "SingleThresholdBS+SingleThresholdIF": "DOC BS+IF",
    "DoubleThresholdBS+SingleThresholdIF": "DOC BS+IF",
    
    "Sequential-SingleThresholdBS+SingleThresholdSPC": "Seq BS+SPC",
    "Sequential-DoubleThresholdBS+DoubleThresholdSPC": "Seq BS+SPC",
    "Sequential-SingleThresholdBS+DoubleThresholdSPC": "Seq BS+SPC",
    "Sequential-DoubleThresholdBS+SingleThresholdSPC": "Seq BS+SPC",
    "Sequential-SingleThresholdBS+SingleThresholdIF": "Seq BS+IF", 
    "Sequential-DoubleThresholdBS+SingleThresholdIF": "Seq BS+IF",
}

best_hyperparameters = {}
best_preprocessing_hyperparameters = {}
best_models = {}

PRFAUC_table_per_method = {}
PRF_mean_table_per_method = {}
PRF_std_table_per_method = {}
avg_fbeta_mean_per_method = {}
avg_fbeta_std_per_method = {}
avg_precision_mean_per_method = {}
avg_precision_std_per_method = {}
avg_recall_mean_per_method = {}
avg_recall_std_per_method = {}

validation_fbeta_per_method = {}

minmax_stats_per_method = {}

for method_name in methods:
    
    #find best preprocessing and method hyperparameters:
    best_model_entry = db_cursor.execute("""
    SELECT e.* 
    FROM experiment_results e 
    WHERE e.metric = (
        SELECT MAX(metric)
        FROM experiment_results
        WHERE method = (?) AND which_split = (?)
    ) AND e.method = (?)
""", (method_name, "Validation", method_name))

    (preprocessing_hash, hyperparameter_hash, _, _, preprocessing_hyperparameter_string_pickle, hyperparameter_string_pickle, validation_metric) = next(best_model_entry)

    best_hyperparameters[method_name] = jsonpickle.decode(hyperparameter_string_pickle, keys=True)
    best_preprocessing_hyperparameters[method_name] = jsonpickle.decode(preprocessing_hyperparameter_string_pickle, keys=True)

    hyperparameters = best_hyperparameters[method_name]     
    model = methods[method_name](model_folder, preprocessing_hash, **hyperparameters)
    best_models[method_name] = model

    PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, method_name, preprocessing_hash)
    PRF_mean_table_path = os.path.join(metric_folder, "PRF_mean_table", which_split, method_name, preprocessing_hash)
    PRF_std_table_path = os.path.join(metric_folder, "PRF_std_table", which_split, method_name, preprocessing_hash)
    avg_fbeta_mean_path = os.path.join(metric_folder, "bootstrap_mean_F"+str(beta), which_split, method_name, preprocessing_hash)
    avg_fbeta_std_path = os.path.join(metric_folder, "bootstrap_std_F"+str(beta), which_split, method_name, preprocessing_hash)
    minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, method_name, preprocessing_hash)
    avg_precision_mean_path = os.path.join(metric_folder, "bootstrap_mean_precision", which_split, method_name, preprocessing_hash)
    avg_precision_std_path = os.path.join(metric_folder, "bootstrap_std_precision", which_split, method_name, preprocessing_hash)
    avg_recall_mean_path = os.path.join(metric_folder, "bootstrap_mean_recall", which_split, method_name, preprocessing_hash)
    avg_recall_std_path = os.path.join(metric_folder, "bootstrap_std_recall", which_split, method_name, preprocessing_hash)
    
    PRFAUC_table_per_method[method_name] = load_table(PRFAUC_table_path, hyperparameter_hash)
    PRF_mean_table_per_method[method_name] = load_table(PRF_mean_table_path, hyperparameter_hash)
    PRF_std_table_per_method[method_name] = load_table(PRF_std_table_path, hyperparameter_hash)
    avg_fbeta_mean_per_method[method_name] = load_metric(avg_fbeta_mean_path, hyperparameter_hash)
    avg_fbeta_std_per_method[method_name] = load_metric(avg_fbeta_std_path, hyperparameter_hash)
    minmax_stats_per_method[method_name] = load_minmax_stats(minmax_stats_path, hyperparameter_hash)
    avg_precision_mean_per_method[method_name] = load_metric(avg_precision_mean_path, hyperparameter_hash)
    avg_precision_std_per_method[method_name] = load_metric(avg_precision_std_path, hyperparameter_hash)
    avg_recall_mean_per_method[method_name] = load_metric(avg_recall_mean_path, hyperparameter_hash)
    avg_recall_std_per_method[method_name] = load_metric(avg_recall_std_path, hyperparameter_hash)
    
    
    validation_fbeta_per_method[method_name] = validation_metric
    
#Make plot of average F score
ordering = {k:i for i, k in enumerate(avg_fbeta_mean_per_method)}
category_names = {method_name:"Average" for method_name in methods}
bootstrapped_Fscore = pd.concat([pd.Series(avg_fbeta_mean_per_method), pd.Series(avg_fbeta_std_per_method), pd.Series(avg_precision_mean_per_method), pd.Series(avg_precision_std_per_method), pd.Series(avg_recall_mean_per_method), pd.Series(avg_recall_std_per_method) , pd.Series(method_groups), pd.Series(validation_fbeta_per_method), pd.Series(ordering), pd.Series(category_names)], axis=1)
bootstrapped_Fscore.columns = ["F1.5 average", "F1.5 stdev", "Precision average", "Precision stdev", "Recall average", "Recall stdev", "Method class", "Validation F1.5", "Ordering", "Length category"]

bootstrapped_Fscore.rename(index=name_abbreviations, inplace=True)

bootstrapped_Fscore = bootstrapped_Fscore.dropna(subset=['Validation F1.5'])
idx_max = bootstrapped_Fscore.groupby('Method class')['Validation F1.5'].idxmax()
# Select the rows with the maximal 'Validation F1.5' for each 'Method class'
average_max_rows = bootstrapped_Fscore.loc[idx_max]
average_max_rows.sort_values(by="Ordering", inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(data=average_max_rows, x=average_max_rows["Method class"], y=average_max_rows['F1.5 average'], hue="Method class", legend=False)
plt.errorbar(x=average_max_rows["Method class"], y=average_max_rows['F1.5 average'], yerr=average_max_rows["F1.5 stdev"], fmt="none", c="k", capsize=20)
plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Method')
plt.ylabel('F1.5 Score (Average)')
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "bootstrap_results.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "bootstrap_results.png"), format="png")
plt.show()


#%%Make plot of F score and AUC per category:
base_plot_df = pd.DataFrame()
for cutoffs in all_cutoffs:
    category = str(cutoffs)
    aucs = {method_name:PRFAUC_table_per_method[method_name]["ROC/AUC"].loc[str(category)] for method_name in methods}
    fbetas = {method_name:PRF_mean_table_per_method[method_name]["F1.5"].loc[str(category)] for method_name in methods}
    fbeta_stds = {method_name:PRF_std_table_per_method[method_name]["F1.5"].loc[str(category)] for method_name in methods}
    recalls = {method_name:PRF_mean_table_per_method[method_name]["recall"].loc[str(category)] for method_name in methods}
    recall_stds = {method_name:PRF_std_table_per_method[method_name]["recall"].loc[str(category)] for method_name in methods}
    precisions = {method_name:PRF_mean_table_per_method[method_name]["precision"].loc[str(category)] for method_name in methods}
    precision_stds = {method_name:PRF_std_table_per_method[method_name]["precision"].loc[str(category)] for method_name in methods}
    
    category_names = {method_name:cutoff_replacement_dict[category] for method_name in methods}
    ordering = {k:i for i, k in enumerate(avg_fbeta_mean_per_method)}
    bootstrapped_Fscore = pd.concat([pd.Series(aucs), pd.Series(fbetas), pd.Series(fbeta_stds), pd.Series(recalls), pd.Series(recall_stds), pd.Series(precisions), pd.Series(precision_stds), pd.Series(method_groups), pd.Series(validation_fbeta_per_method), pd.Series(ordering), pd.Series(category_names)], axis=1)
    bootstrapped_Fscore.columns = ["AUC", "F1.5 average", "F1.5 stdev", "Recall average", "Recall stdev", "Precision average", "Precision stdev", "Method class", "Validation F1.5", "Ordering", "Length category"]
    
    bootstrapped_Fscore.rename(index=name_abbreviations, inplace=True)
    
    bootstrapped_Fscore = bootstrapped_Fscore.dropna(subset=['Validation F1.5'])
    idx_max = bootstrapped_Fscore.groupby('Method class')['Validation F1.5'].idxmax()
    # Select the rows with the maximal 'Validation F1.5' for each 'Method class'
    max_rows = bootstrapped_Fscore.loc[idx_max]
    max_rows.sort_values(by="Ordering", inplace=True)

    base_plot_df = pd.concat([base_plot_df, max_rows])

F_score_base_plot_df = pd.concat([base_plot_df, average_max_rows])


plt.figure(figsize=(10,6))
sns.barplot(data=base_plot_df, x=base_plot_df["Method class"], y=base_plot_df['AUC'], hue="Length category")

ax = plt.gca()
bars = ax.patches

plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Method')
plt.ylabel('AUC-ROC')
plt.tight_layout()

plt.legend()
plt.savefig(os.path.join(figure_folder, "AUC_results_per_category.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "AUC_results_per_category.png"), format="png")



plt.show()


plt.figure(figsize=(10,6))
sns.barplot(data=F_score_base_plot_df, x=F_score_base_plot_df["Method class"], y=F_score_base_plot_df['F1.5 average'], hue="Length category")

ax = plt.gca()
bars = ax.patches

# Calculate the x-values of the center of each bar
bar_centers = [(bar.get_x() + bar.get_width() / 2) for bar in bars]
#Only get the first X bar centers, after that are dummy values
plt.errorbar(x=bar_centers[:len(F_score_base_plot_df['F1.5 average'])], y=F_score_base_plot_df['F1.5 average'], yerr=F_score_base_plot_df["F1.5 stdev"], fmt="none", c="k", capsize=4)


plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Method')
plt.ylabel('F1.5 Score (Average)')
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "bootstrap_results_per_category.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "bootstrap_results_per_category.png"), format="png")



plt.show()

plt.figure(figsize=(10,6))
sns.barplot(data=F_score_base_plot_df, x=F_score_base_plot_df["Method class"], y=F_score_base_plot_df['Recall average'], hue="Length category")

ax = plt.gca()
bars = ax.patches

# Calculate the x-values of the center of each bar
bar_centers = [(bar.get_x() + bar.get_width() / 2) for bar in bars]
#Only get the first X bar centers, after that are dummy values
plt.errorbar(x=bar_centers[:len(F_score_base_plot_df['Recall average'])], y=F_score_base_plot_df['Recall average'], yerr=F_score_base_plot_df["Recall stdev"], fmt="none", c="k", capsize=4)


plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Method')
plt.ylabel('Recall (Average)')
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "recall_bootstrap_results_per_category.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "recall_bootstrap_results_per_category.png"), format="png")

plt.show()

plt.figure(figsize=(10,6))
sns.barplot(data=F_score_base_plot_df, x=F_score_base_plot_df["Method class"], y=F_score_base_plot_df['Precision average'], hue="Length category")

ax = plt.gca()
bars = ax.patches

# Calculate the x-values of the center of each bar
bar_centers = [(bar.get_x() + bar.get_width() / 2) for bar in bars]
#Only get the first X bar centers, after that are dummy values
plt.errorbar(x=bar_centers[:len(F_score_base_plot_df['Precision average'])], y=F_score_base_plot_df['Precision average'], yerr=F_score_base_plot_df["Precision stdev"], fmt="none", c="k", capsize=4)


plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Method')
plt.ylabel('Precision (Average)')
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "precision_bootstrap_results_per_category.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "precision_bootstrap_results_per_category.png"), format="png")

plt.show()

#%% combine previous plots:


fig, axes = plt.subplots(3, 1, figsize=(12, 14.3), sharex=True)

sns.barplot(data=F_score_base_plot_df, x=F_score_base_plot_df["Method class"], y=F_score_base_plot_df['F1.5 average'], hue="Length category", ax=axes[0])

bars = axes[0].patches

# Calculate the x-values of the center of each bar
bar_centers = [(bar.get_x() + bar.get_width() / 2) for bar in bars]
#Only get the first X bar centers, after that are dummy values

axes[0].errorbar(x=bar_centers[:len(F_score_base_plot_df['F1.5 average'])], y=F_score_base_plot_df['F1.5 average'], yerr=F_score_base_plot_df["F1.5 stdev"], fmt="none", c="k", capsize=4)
axes[0].set_ylabel('F1.5 Score (Average)', fontsize=18)

sns.barplot(data=F_score_base_plot_df, x=F_score_base_plot_df["Method class"], y=F_score_base_plot_df['Recall average'], hue="Length category", ax=axes[1])

axes[0].tick_params(axis='y', labelsize=18)

bars = axes[1].patches

# Calculate the x-values of the center of each bar
bar_centers = [(bar.get_x() + bar.get_width() / 2) for bar in bars]
#Only get the first X bar centers, after that are dummy values

axes[1].errorbar(x=bar_centers[:len(F_score_base_plot_df['Recall average'])], y=F_score_base_plot_df['Recall average'], yerr=F_score_base_plot_df["Recall stdev"], fmt="none", c="k", capsize=4)
axes[1].set_ylabel('Recall (Average)', fontsize=18)

sns.barplot(data=F_score_base_plot_df, x=F_score_base_plot_df["Method class"], y=F_score_base_plot_df['Precision average'], hue="Length category", ax=axes[2])

axes[1].tick_params(axis='y', labelsize=18)

bars = axes[2].patches

# Calculate the x-values of the center of each bar
bar_centers = [(bar.get_x() + bar.get_width() / 2) for bar in bars]
#Only get the first X bar centers, after that are dummy values

axes[2].errorbar(x=bar_centers[:len(F_score_base_plot_df['Precision average'])], y=F_score_base_plot_df['Precision average'], yerr=F_score_base_plot_df["Precision stdev"], fmt="none", c="k", capsize=4)
axes[2].set_ylabel('Precision (Average)', fontsize=18)

# Rotate x-axis labels and add a common x-axis label
#plt.setp(axes, xticks=range(len(base_plot_df["Method class"].unique())), xticklabels=base_plot_df["Method class"].unique(), xticksrotation=90)
axes[2].set_xticks(range(len(F_score_base_plot_df["Method class"].unique())))
axes[2].set_xticklabels(F_score_base_plot_df["Method class"].unique(), rotation=90, fontsize=18)
axes[2].set_xlabel("Method Class", fontsize=25)

axes[2].tick_params(axis='y', labelsize=18)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Move the legend to the top of the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(F_score_base_plot_df["Length category"].unique()), bbox_to_anchor=(0.5, 0.98), fontsize=20)
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].get_legend().remove()

plt.savefig(os.path.join(figure_folder, "combined_bootstrap_results_per_category.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "combined_bootstrap_results_per_category.png"), format="png")

plt.show()
#%% visualize minmax stats
from bidict import bidict
#subset dict so we only get stats for best performing method on validation:
    
ordering = {k:i for i, k in enumerate(avg_fbeta_mean_per_method)}
category_names = {method_name:"Average" for method_name in methods}
bootstrapped_Fscore = pd.concat([pd.Series(avg_fbeta_mean_per_method), pd.Series(avg_fbeta_std_per_method), pd.Series(method_groups), pd.Series(validation_fbeta_per_method), pd.Series(ordering), pd.Series(category_names)], axis=1)
bootstrapped_Fscore.columns = ["F1.5 average", "F1.5 stdev", "Method class", "Validation F1.5", "Ordering", "Length category"]

bootstrapped_Fscore.rename(index=name_abbreviations, inplace=True)

bootstrapped_Fscore = bootstrapped_Fscore.dropna(subset=['Validation F1.5'])
idx_max = bootstrapped_Fscore.groupby('Method class')['Validation F1.5'].idxmax()
# Select the rows with the maximal 'Validation F1.5' for each 'Method class'
average_max_rows = bootstrapped_Fscore.loc[idx_max]
average_max_rows.sort_values(by="Ordering", inplace=True)

best_minmax_dict = {}
best_hyperparameter_per_method_group = {}
best_model_per_method_group = {}

abbrev_bidict = bidict(name_abbreviations)


for method_name in list(average_max_rows["Method class"].index):
    best_minmax_dict[method_groups[abbrev_bidict.inverse[method_name]]] = minmax_stats_per_method[abbrev_bidict.inverse[method_name]]
    best_hyperparameter_per_method_group[method_groups[abbrev_bidict.inverse[method_name]]] = best_hyperparameters[abbrev_bidict.inverse[method_name]]
    best_model_per_method_group[method_groups[abbrev_bidict.inverse[method_name]]] = best_models[abbrev_bidict.inverse[method_name]]
    

Seq_best_df = best_minmax_dict["Seq BS+SPC"]/1000
BS_best_df = best_minmax_dict["BS"]/1000
SPC_best_df = best_minmax_dict["SPC"]/1000

# Creating a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plotting each dataframe on a separate subplot

#For no filtering plot we can use whatever method, as unfiltered mins and maxs are the same
sns.scatterplot(data=Seq_best_df, x="X_maxs", y="X_maxs_no_filter", ax=axes[0, 0], s=60)
axes[0, 0].set_xlim(left=0)
axes[0, 0].set_ylim(bottom=-1)
axes[0, 0].set_xlabel("Ground truth maximum load (MW)", fontsize=15)
axes[0, 0].set_ylabel("Predicted maximum load (MW)", fontsize=15)
axes[0, 0].set_aspect('equal', adjustable='box')
axes[0, 0].set_title("Unfiltered", fontsize=18)

axes[0, 0].tick_params(axis='x', labelsize=15)
axes[0, 0].tick_params(axis='y', labelsize=15)

sns.scatterplot(data=BS_best_df, x="X_maxs", y="X_pred_maxs", ax=axes[0, 1], s=60)
axes[0, 1].set_xlim(left=0)
axes[0, 1].set_ylim(bottom=-1)
axes[0, 1].set_xlabel("Ground truth maximum load (MW)", fontsize=15)
axes[0, 1].set_ylabel("Predicted maximum load (MW)", fontsize=15)
axes[0, 1].set_aspect('equal', adjustable='box')
axes[0, 1].set_title("BS", fontsize=18)

axes[0, 1].tick_params(axis='x', labelsize=15)
axes[0, 1].tick_params(axis='y', labelsize=15)

sns.scatterplot(data=SPC_best_df, x="X_maxs", y="X_pred_maxs", ax=axes[1, 0], s=60)
axes[1, 0].set_xlim(left=0)
axes[1, 0].set_ylim(bottom=-1)
axes[1, 0].set_xlabel("Ground truth maximum load (MW)", fontsize=15)
axes[1, 0].set_ylabel("Predicted maximum load (MW)", fontsize=15)
axes[1, 0].set_aspect('equal', adjustable='box')
axes[1, 0].set_title("SPC", fontsize=18)

axes[1, 0].tick_params(axis='x', labelsize=15)
axes[1, 0].tick_params(axis='y', labelsize=15)

sns.scatterplot(data=Seq_best_df, x="X_maxs", y="X_pred_maxs", ax=axes[1, 1], s=60)
axes[1, 1].set_xlim(left=0)
axes[1, 1].set_ylim(bottom=-1)
axes[1, 1].set_xlabel("Ground truth maximum load (MW)", fontsize=15)
axes[1, 1].set_ylabel("Predicted maximum load (MW)", fontsize=15)
axes[1, 1].set_aspect('equal', adjustable='box')
axes[1, 1].set_title("Seq BS+SPC", fontsize=18)

axes[1, 1].tick_params(axis='x', labelsize=15)
axes[1, 1].tick_params(axis='y', labelsize=15)


plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "maximum_load_estimates.png"), format="png")
plt.savefig(os.path.join(figure_folder, "maximum_load_estimates.pdf"), format="pdf")
plt.show()

#%%


Seq_best_df = best_minmax_dict["Seq BS+SPC"]
BS_best_df = best_minmax_dict["BS"]
SPC_best_df = best_minmax_dict["SPC"]

Seq_min_best_df = Seq_best_df.loc[Seq_best_df["has_negative_load"]]/1000
BS_min_best_df = BS_best_df.loc[BS_best_df["has_negative_load"]]/1000
SPC_min_best_df = SPC_best_df.loc[SPC_best_df["has_negative_load"]]/1000

# Creating a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plotting each dataframe on a separate subplot

#For no filtering plot we can use whatever method, as unfiltered mins and maxs are the same
sns.scatterplot(data=Seq_min_best_df, x="X_mins", y="X_mins_no_filter", ax=axes[0, 0], s=60)
axes[0, 0].set_xlim(right=0)
axes[0, 0].set_ylim(top=1)
axes[0, 0].set_xlabel("Ground truth minimum load (MW)", fontsize=15)
axes[0, 0].set_ylabel("Predicted minimum load (MW)", fontsize=15)
axes[0, 0].set_aspect('equal', adjustable='box')
axes[0, 0].set_title("Unfiltered", fontsize=18)

axes[0, 0].tick_params(axis='x', labelsize=15)
axes[0, 0].tick_params(axis='y', labelsize=15)

sns.scatterplot(data=Seq_min_best_df, x="X_mins", y="X_pred_mins", ax=axes[0, 1], s=60)
axes[0, 1].set_xlim(right=0)
axes[0, 1].set_ylim(top=1)
axes[0, 1].set_xlabel("Ground truth minimum load (MW)", fontsize=15)
axes[0, 1].set_ylabel("Predicted minimum load (MW)", fontsize=15)
axes[0, 1].set_aspect('equal', adjustable='box')
axes[0, 1].set_title("BS", fontsize=18)

axes[0, 1].tick_params(axis='x', labelsize=15)
axes[0, 1].tick_params(axis='y', labelsize=15)

sns.scatterplot(data=Seq_min_best_df, x="X_mins", y="X_pred_mins", ax=axes[1, 0], s=60)
axes[1, 0].set_xlim(right=0)
axes[1, 0].set_ylim(top=1)
axes[1, 0].set_xlabel("Ground truth minimum load (MW)", fontsize=15)
axes[1, 0].set_ylabel("Predicted minimum load (MW)", fontsize=15)
axes[1, 0].set_aspect('equal', adjustable='box')
axes[1, 0].set_title("SPC", fontsize=18)

axes[1, 0].tick_params(axis='x', labelsize=15)
axes[1, 0].tick_params(axis='y', labelsize=15)

sns.scatterplot(data=Seq_min_best_df, x="X_mins", y="X_pred_mins", ax=axes[1, 1], s=60)
axes[1, 1].set_xlim(right=0)
axes[1, 1].set_ylim(top=1)
axes[1, 1].set_xlabel("Ground truth minimum load (MW)", fontsize=15)
axes[1, 1].set_ylabel("Predicted minimum load (MW)", fontsize=15)
axes[1, 1].set_aspect('equal', adjustable='box')
axes[1, 1].set_title("Seq BS+SPC", fontsize=18)

axes[1, 1].tick_params(axis='x', labelsize=15)
axes[1, 1].tick_params(axis='y', labelsize=15)

plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "minimum_load_estimates.png"), format="png")
plt.savefig(os.path.join(figure_folder, "minimum_load_estimates.pdf"), format="pdf")
plt.show()


#%% print prediction stats:
best_df = best_minmax_dict["Seq BS+SPC"]

acceptable_margin = 10 #percentage the predictions need to be in

percentage_perfect_min = np.sum(Seq_min_best_df["X_mins"] == Seq_min_best_df["X_pred_mins"])/Seq_min_best_df.shape[0]*100
percentage_acceptable_min = np.sum(np.abs(Seq_min_best_df["min_differences"]/Seq_min_best_df["X_mins"]*100) < acceptable_margin)/Seq_min_best_df.shape[0]*100

percentage_perfect_max = np.sum(Seq_best_df["X_maxs"] == Seq_best_df["X_pred_maxs"])/Seq_best_df.shape[0]*100
percentage_acceptable_max = np.sum(np.abs(Seq_best_df["max_differences"]/Seq_best_df["X_maxs"]*100) < acceptable_margin)/Seq_best_df.shape[0]*100


print("The maximum load predictions are perfect in {0:.2f}% of all cases".format(percentage_perfect_max))
print("The maximum load predictions are within a {0}% error margin in {1:.2f}% of all cases".format(acceptable_margin, percentage_acceptable_max))
print("")
print("The minimum load predictions are perfect in {0:.2f}% of all cases".format(percentage_perfect_min))
print("The minimum load predictions are within a {0}% error margin in {1:.2f}% of all cases".format(acceptable_margin, percentage_acceptable_min))

#%% construct best hyperparameter table:
    
hyperparameter_translation_dict = {"n_estimators":"$n_{\\textrm{estimators}}$",
                                   "quantiles":"($q_{\\textrm{lower}}\\%$, $q_{\\textrm{upper}}\\%$)",
                                   "beta":"$\\beta$",
                                   "model":None,
                                   "min_size":"$l$",
                                   "jump":"$j$",
                                   "reference_point":"$\\textit{reference\\_point}$",
                                   "forest_per_station":None,
                                   "scaling":None,
                                   "score_function_kwargs":None,
                                   "penalty":"$C$",
                                   "threshold_strategy":"Threshold strategy"}

thresholds_strategy_translation_dict = {"DoubleThreshold":"Asymmetrical",
                                        "SingleThreshold":"Symmetrical"}

hyperparameter_list = []

column_names = ["Ensemble method", "Combination", "Method", "Hyperparameter", "Hyperparameter values"]

for method_name in best_hyperparameter_per_method_group:
    model = best_model_per_method_group[method_name]
    
    if "method_hyperparameter_dict_list" in best_hyperparameter_per_method_group[method_name].keys():
        print("is ensemble")
        #if it is sequential ensemble:
        if "segmentation_method" in best_hyperparameter_per_method_group[method_name].keys():
            print("is sequential ensemble")
            
            #First do segmenter:
            for hyperparameter in best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][0]:
                
                
                hyperparameter_name = hyperparameter_translation_dict[hyperparameter]
                if hyperparameter_name is not None:
                    ensemble_name = "Sequential"
                    method_name_table = method_groups[model.segmentation_method.method_name]
                    row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Seq ", ""), column_names[2]:method_name_table, column_names[3]:hyperparameter_name, column_names[4]:best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][0][hyperparameter]}
                    hyperparameter_list.append(row)
                    
            #Additionally, append threshold strategy and optimal values:
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Seq ", ""), column_names[2]:method_name_table, column_names[3]:"Threshold strategy", column_names[4]:thresholds_strategy_translation_dict[model.segmentation_method.threshold_optimization_method]}
            hyperparameter_list.append(row)
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Seq ", ""), column_names[2]:method_name_table, column_names[3]:"Optimal threshold(s)", column_names[4]:model.segmentation_method.optimal_threshold}
            hyperparameter_list.append(row)
            
            #then do anomaly detector:
            for hyperparameter in best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][1]:
                
                hyperparameter_name = hyperparameter_translation_dict[hyperparameter]
                if hyperparameter_name is not None:
                    ensemble_name = "Sequential"
                    method_name_table = method_groups[model.anomaly_detection_method.method_name]
                    row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Seq ", ""), column_names[2]:method_name_table, column_names[3]:hyperparameter_name, column_names[4]:best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][1][hyperparameter]}
                    hyperparameter_list.append(row)
                    
            #Additionally, append threshold strategy and optimal values:
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Seq ", ""), column_names[2]:method_name_table, column_names[3]:"Threshold strategy", column_names[4]:thresholds_strategy_translation_dict[model.anomaly_detection_method.threshold_optimization_method]}
            hyperparameter_list.append(row)
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Seq ", ""), column_names[2]:method_name_table, column_names[3]:"Optimal threshold(s)", column_names[4]:model.anomaly_detection_method.optimal_threshold}
            hyperparameter_list.append(row)
            
        #DOC ensemble:
        elif "cutoffs_per_method" in best_hyperparameter_per_method_group[method_name].keys():
            print("is DOC ensemble")
            
            #First do segmenter:
            for hyperparameter in best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][0]:
                
                
                hyperparameter_name = hyperparameter_translation_dict[hyperparameter]
                if hyperparameter_name is not None:
                    ensemble_name = "DOC"
                    method_name_table = method_groups[model.models[0].method_name]
                    row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("DOC ", ""), column_names[2]:method_name_table, column_names[3]:hyperparameter_name, column_names[4]:best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][0][hyperparameter]}
                    hyperparameter_list.append(row)
                    
            #Additionally, append threshold strategy and optimal values:
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("DOC ", ""), column_names[2]:method_name_table, column_names[3]:"Threshold strategy", column_names[4]:thresholds_strategy_translation_dict[model.models[0].threshold_optimization_method]}
            hyperparameter_list.append(row)
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("DOC ", ""), column_names[2]:method_name_table, column_names[3]:"Optimal threshold(s)", column_names[4]:model.models[0].optimal_threshold}
            hyperparameter_list.append(row)
            
            #then do anomaly detector:
            for hyperparameter in best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][1]:
                
                hyperparameter_name = hyperparameter_translation_dict[hyperparameter]
                if hyperparameter_name is not None:
                    ensemble_name = "DOC"
                    method_name_table = method_groups[model.models[1].method_name]
                    row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("DOC ", ""), column_names[2]:method_name_table, column_names[3]:hyperparameter_name, column_names[4]:best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][1][hyperparameter]}
                    hyperparameter_list.append(row)
                    
            #Additionally, append threshold strategy and optimal values:
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("DOC ", ""), column_names[2]:method_name_table, column_names[3]:"Threshold strategy", column_names[4]:thresholds_strategy_translation_dict[model.models[1].threshold_optimization_method]}
            hyperparameter_list.append(row)
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("DOC ", ""), column_names[2]:method_name_table, column_names[3]:"Optimal threshold(s)", column_names[4]:model.models[1].optimal_threshold}
            hyperparameter_list.append(row)
            
        elif "all_cutoffs" in best_hyperparameter_per_method_group[method_name].keys():
            print("is naive ensemble")
            
            #First do segmenter:
            for hyperparameter in best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][0]:
                
                
                hyperparameter_name = hyperparameter_translation_dict[hyperparameter]
                if hyperparameter_name is not None:
                    ensemble_name = "Naive"
                    method_name_table = method_groups[model.models[0].method_name]
                    row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Naive ", ""), column_names[2]:method_name_table, column_names[3]:hyperparameter_name, column_names[4]:best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][0][hyperparameter]}
                    hyperparameter_list.append(row)
                    
            #Additionally, append threshold strategy and optimal values:
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Naive ", ""), column_names[2]:method_name_table, column_names[3]:"Threshold strategy", column_names[4]:thresholds_strategy_translation_dict[model.models[0].threshold_optimization_method]}
            hyperparameter_list.append(row)
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Naive ", ""), column_names[2]:method_name_table, column_names[3]:"Optimal threshold(s)", column_names[4]:model.models[0].optimal_threshold}
            hyperparameter_list.append(row)
            
            #then do anomaly detector:
            for hyperparameter in best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][1]:
                
                hyperparameter_name = hyperparameter_translation_dict[hyperparameter]
                if hyperparameter_name is not None:
                    ensemble_name = "Naive"
                    method_name_table = method_groups[model.models[1].method_name]
                    row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Naive ", ""), column_names[2]:method_name_table, column_names[3]:hyperparameter_name, column_names[4]:best_hyperparameter_per_method_group[method_name]["method_hyperparameter_dict_list"][1][hyperparameter]}
                    hyperparameter_list.append(row)
                    
            #Additionally, append threshold strategy and optimal values:
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Naive ", ""), column_names[2]:method_name_table, column_names[3]:"Threshold strategy", column_names[4]:thresholds_strategy_translation_dict[model.models[1].threshold_optimization_method]}
            hyperparameter_list.append(row)
            row = {column_names[0]:ensemble_name, column_names[1]:method_name.replace("Naive ", ""), column_names[2]:method_name_table, column_names[3]:"Optimal threshold(s)", column_names[4]:model.models[1].optimal_threshold}
            hyperparameter_list.append(row)
            
        else:
            ValueError("Ensemble method does not have valid API, is it a DOC, Naive or Sequential ensemble?")
    else:
        print("is not ensemble")
        for hyperparameter in best_hyperparameter_per_method_group[method_name]:
            
            
            hyperparameter_name = hyperparameter_translation_dict[hyperparameter]
            if hyperparameter_name is not None:
                ensemble_name = "No ensemble"
                method_name_table = method_name
                row = {column_names[0]:ensemble_name, column_names[1]:"-", column_names[2]:method_name_table, column_names[3]:hyperparameter_name, column_names[4]:best_hyperparameter_per_method_group[method_name][hyperparameter]}
                hyperparameter_list.append(row)
                
        #Additionally, append threshold strategy and optimal values:
        row = {column_names[0]:ensemble_name, column_names[1]:"-", column_names[2]:method_name_table, column_names[3]:"Threshold strategy", column_names[4]:thresholds_strategy_translation_dict[model.threshold_optimization_method]}
        hyperparameter_list.append(row)
        row = {column_names[0]:ensemble_name, column_names[1]:"-", column_names[2]:method_name_table, column_names[3]:"Optimal threshold(s)", column_names[4]:model.optimal_threshold}
        hyperparameter_list.append(row)


hyperparameter_df = pd.DataFrame(hyperparameter_list, columns=column_names)

hyperparameter_df_index = pd.MultiIndex.from_frame(hyperparameter_df[["Ensemble method", "Combination", "Method", "Hyperparameter"]])

hyperparameter_df = pd.DataFrame(hyperparameter_df["Hyperparameter values"])
hyperparameter_df.index = hyperparameter_df_index

hyperparameter_df.loc[["No ensemble", "Naive"]].to_latex(buf=os.path.join(table_folder, "best_hyperparameters_first_half.tex"), escape=False, multirow=True)

hyperparameter_df.loc[["DOC", "Sequential"]].to_latex(buf=os.path.join(table_folder, "best_hyperparameters_second_half.tex"), escape=False, multirow=True)