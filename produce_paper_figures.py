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

from hashlib import sha256


from src.plot_functions import plot_S_original, plot_BU_original, plot_predictions
from src.io_functions import load_table, load_batch, load_dataframe_list, load_metric
from src.reporting_functions import bootstrap_stats_to_printable

from src.methods import SingleThresholdStatisticalProcessControl
from src.methods import DoubleThresholdStatisticalProcessControl
from src.methods import SingleThresholdIsolationForest

from src.methods import SingleThresholdBinarySegmentation
from src.methods import DoubleThresholdBinarySegmentation

from src.methods import StackEnsemble
from src.methods import NaiveStackEnsemble
from src.methods import SequentialEnsemble

from src.preprocess import preprocess_per_batch_and_write

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
cutoff_replacement_dict = {"(0, 24)":"15m-8h", "(24, 288)":"8h-3d","(288, 4032)":"3d-42d", "(4032, inf)":"42d and longer"}

#%% Visualize/tabularize input data and preprocessing

#measurement_example.pdf
station_ID = "001"

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

plt.axhline(y=np.max(X_df["S_original"]), color='black', linestyle='dashed', label="Maximum load")
plt.axhline(y=np.min(X_df["S_original"]), color='black', linestyle='dotted', label="Minimum load")

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
plt.ylabel("Counts (log)")
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
from src.methods import SingleThresholdStatisticalProcessControl
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
plt.ylabel(r"F$1.5_{average}$")
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

PRF_mean_table_per_method = {}
PRF_std_table_per_method = {}
avg_fbeta_mean_per_method = {}
avg_fbeta_std_per_method = {}

validation_fbeta_per_method = {}

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

    PRF_mean_table_path = os.path.join(metric_folder, "PRF_mean_table", which_split, method_name, preprocessing_hash)
    PRF_std_table_path = os.path.join(metric_folder, "PRF_std_table", which_split, method_name, preprocessing_hash)
    avg_fbeta_mean_path = os.path.join(metric_folder, "bootstrap_mean_F"+str(beta), which_split, method_name, preprocessing_hash)
    avg_fbeta_std_path = os.path.join(metric_folder, "bootstrap_std_F"+str(beta), which_split, method_name, preprocessing_hash)
    
    PRF_mean_table_per_method[method_name] = load_table(PRF_mean_table_path, hyperparameter_hash)
    PRF_std_table_per_method[method_name] = load_table(PRF_std_table_path, hyperparameter_hash)
    avg_fbeta_mean_per_method[method_name] = load_metric(avg_fbeta_mean_path, hyperparameter_hash)
    avg_fbeta_std_per_method[method_name] = load_metric(avg_fbeta_std_path, hyperparameter_hash)
    
    validation_fbeta_per_method[method_name] = validation_metric
    
#Make plot of average F score
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


#%%Make plot of F score per category:
base_plot_df = pd.DataFrame()
for cutoffs in all_cutoffs:
    category = str(cutoffs)
    fbetas = {method_name:PRF_mean_table_per_method[method_name]["F1.5"].loc[str(category)] for method_name in methods}
    fbeta_stds = {method_name:PRF_std_table_per_method[method_name]["F1.5"].loc[str(category)] for method_name in methods}
    category_names = {method_name:cutoff_replacement_dict[category] for method_name in methods}
    ordering = {k:i for i, k in enumerate(avg_fbeta_mean_per_method)}
    bootstrapped_Fscore = pd.concat([pd.Series(fbetas), pd.Series(fbeta_stds), pd.Series(method_groups), pd.Series(validation_fbeta_per_method), pd.Series(ordering), pd.Series(category_names)], axis=1)
    bootstrapped_Fscore.columns = ["F1.5 average", "F1.5 stdev", "Method class", "Validation F1.5", "Ordering", "Length category"]
    
    bootstrapped_Fscore.rename(index=name_abbreviations, inplace=True)
    
    bootstrapped_Fscore = bootstrapped_Fscore.dropna(subset=['Validation F1.5'])
    idx_max = bootstrapped_Fscore.groupby('Method class')['Validation F1.5'].idxmax()
    # Select the rows with the maximal 'Validation F1.5' for each 'Method class'
    max_rows = bootstrapped_Fscore.loc[idx_max]
    max_rows.sort_values(by="Ordering", inplace=True)

    base_plot_df = pd.concat([base_plot_df, max_rows])
base_plot_df = pd.concat([base_plot_df, average_max_rows])


plt.figure(figsize=(10,6))
sns.barplot(data=base_plot_df, x=base_plot_df["Method class"], y=base_plot_df['F1.5 average'], hue="Length category")

ax = plt.gca()
bars = ax.patches

# Calculate the x-values of the center of each bar
bar_centers = [(bar.get_x() + bar.get_width() / 2) for bar in bars]
#Only get the first X bar centers, after that are dummy values
plt.errorbar(x=bar_centers[:len(base_plot_df['F1.5 average'])], y=base_plot_df['F1.5 average'], yerr=base_plot_df["F1.5 stdev"], fmt="none", c="k", capsize=4)


plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Method')
plt.ylabel('F1.5 Score (Average)')
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "bootstrap_results_per_category.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "bootstrap_results_per_category.png"), format="png")

plt.show()

#%% Plot recall per category:
    
base_plot_df = pd.DataFrame()
for cutoffs in all_cutoffs:
    category = str(cutoffs)
    recalls = {method_name:PRF_mean_table_per_method[method_name]["recall"].loc[str(category)] for method_name in methods}
    recall_stds = {method_name:PRF_std_table_per_method[method_name]["recall"].loc[str(category)] for method_name in methods}
    category_names = {method_name:cutoff_replacement_dict[category] for method_name in methods}
    ordering = {k:i for i, k in enumerate(avg_fbeta_mean_per_method)}
    bootstrapped_Fscore = pd.concat([pd.Series(recalls), pd.Series(recall_stds), pd.Series(method_groups), pd.Series(validation_fbeta_per_method), pd.Series(ordering), pd.Series(category_names)], axis=1)
    bootstrapped_Fscore.columns = ["Recall average", "Recall stdev", "Method class", "Validation F1.5", "Ordering", "Length category"]
    
    bootstrapped_Fscore.rename(index=name_abbreviations, inplace=True)
    
    bootstrapped_Fscore = bootstrapped_Fscore.dropna(subset=['Validation F1.5'])
    idx_max = bootstrapped_Fscore.groupby('Method class')['Validation F1.5'].idxmax()
    # Select the rows with the maximal 'Validation F1.5' for each 'Method class'
    max_rows = bootstrapped_Fscore.loc[idx_max]
    max_rows.sort_values(by="Ordering", inplace=True)

    base_plot_df = pd.concat([base_plot_df, max_rows])
#base_plot_df = pd.concat([base_plot_df, average_max_rows])


plt.figure(figsize=(10,6))
sns.barplot(data=base_plot_df, x=base_plot_df["Method class"], y=base_plot_df['Recall average'], hue="Length category")

ax = plt.gca()
bars = ax.patches

# Calculate the x-values of the center of each bar
bar_centers = [(bar.get_x() + bar.get_width() / 2) for bar in bars]
#Only get the first X bar centers, after that are dummy values
plt.errorbar(x=bar_centers[:len(base_plot_df['Recall average'])], y=base_plot_df['Recall average'], yerr=base_plot_df["Recall stdev"], fmt="none", c="k", capsize=4)


plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Method')
plt.ylabel('Recall (Average)')
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "recall_bootstrap_results_per_category.pdf"), format="pdf")
plt.savefig(os.path.join(figure_folder, "recall_bootstrap_results_per_category.png"), format="png")

plt.show()