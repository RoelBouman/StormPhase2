#%% package loading
import os
import pickle
import jsonpickle
import sqlite3

import pandas as pd
import numpy as np

from hashlib import sha256

import seaborn as sns

from src.plot_functions import plot_predictions

from src.methods import SingleThresholdStatisticalProcessControl
from src.methods import DoubleThresholdStatisticalProcessControl
from src.methods import SingleThresholdIsolationForest

from src.methods import SingleThresholdBinarySegmentation
from src.methods import DoubleThresholdBinarySegmentation

from src.evaluation import f_beta

sns.set()

#%% Data loading

dataset = "OS_data" #alternatively: route_data
data_folder = os.path.join("data", dataset)
result_folder = os.path.join("results", dataset)
intermediates_folder = os.path.join("intermediates", dataset)
model_folder = os.path.join("saved_models", dataset)

table_folder = "Tables"
figure_folder = "Figures"

os.makedirs(table_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")
metric_folder = os.path.join(result_folder, "metrics")

preprocessed_folder = os.path.join(intermediates_folder, "preprocessed_data_csvs")

train_name = "Train"
test_name = "Test"
validation_name = "Validation"

all_dataset_names = [train_name, test_name, validation_name]

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

#%% connect to database

DBFILE = dataset+"_experiment_results.db"
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()

#%% choose station IDs

station_IDs = ["1","041"]

train_IDs = sorted(os.listdir(os.path.join(data_folder, "Train", "X")))
test_IDs = sorted(os.listdir(os.path.join(data_folder, "Test", "X")))
validation_IDs = sorted(os.listdir(os.path.join(data_folder, "Validation", "X")))

station_ID_dict = {"Train":train_IDs, "Test":test_IDs, "Validation":validation_IDs}

all_station_IDs = train_IDs + test_IDs + validation_IDs

train_ID_dict = {ID.replace(".csv", ""): "Train" for ID in train_IDs}
test_ID_dict = {ID.replace(".csv", ""): "Test" for ID in test_IDs}
validation_ID_dict = {ID.replace(".csv", ""): "Validation" for ID in validation_IDs}

#fastest dict merge: https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one
station_dataset_dict = dict(train_ID_dict, **test_ID_dict)
station_dataset_dict.update(validation_ID_dict)

#%% choose HP

use_best_model = True
method_name = "SingleThresholdBS"

# if use_best_model is False, use these
preprocessing_hyperparameters = {'lin_fit_quantiles': (10, 90), 'subsequent_nr': 5}
#model_hyperparameters = {'quantiles': (10, 90), 'score_function_kwargs': {'beta': 1.5}}
#model_hyperparameters = {'forest_per_station': False, 'n_estimators': 1000, 'quantiles': (20, 80), 'scaling': True, 'score_function_kwargs': {'beta': 1.5}}
model_hyperparameters = {'beta': 0.015, 'jump': 10, 'min_size': 50, 'model': 'l1', 'penalty': 'fused_lasso', 'quantiles': (15, 85), 'reference_point': 'mean', 'scaling': True, 'score_function_kwargs': {'beta': 1.5}}

#%% hyperparameter selection

# use your own hyperparameters
if not use_best_model:
    # will throw error if combination of HP has not been run in main yet  
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()

# use best model hyperparameters
else:    
    best_model_entry = db_cursor.execute("SELECT e.* FROM experiment_results e WHERE e.metric = (SELECT MAX(metric)FROM experiment_results WHERE method = (?) AND which_split = (?))", (method_name, "Validation"))
    (preprocessing_hash, hyperparameter_hash, _, _, _, _, _) = next(best_model_entry)
    
    db_result = db_cursor.execute("SELECT method_hyperparameters FROM experiment_results WHERE preprocessing_hash='{}' AND hyperparameter_hash='{}'".format(preprocessing_hash, hyperparameter_hash)).fetchone()[0]
    model_hyperparameters = jsonpickle.loads(db_result)


#%% load model

beta = 1.5
def score_function(precision, recall):
    return f_beta(precision, recall, beta)

#model = SingleThresholdStatisticalProcessControl(model_folder, preprocessing_hash, **model_hyperparameters, score_function=score_function)
#model = SingleThresholdIsolationForest(model_folder, preprocessing_hash, **model_hyperparameters, score_function=score_function)
# model = SingleThresholdBinarySegmentation(model_folder, preprocessing_hash, **model_hyperparameters, score_function=score_function)
model = DoubleThresholdBinarySegmentation(model_folder, preprocessing_hash, **model_hyperparameters, score_function=score_function)

# get hash (if not using best model) for prediction loading
hyperparameter_hash = model.get_hyperparameter_hash()

#%% load preprocessed X dfs

X_dfs = []

for station_ID in station_IDs:
    X_df = pd.read_csv(os.path.join(preprocessed_folder, station_dataset_dict[station_ID], preprocessing_hash, station_ID + ".csv"))
    
    X_dfs.append(X_df)

#%% load y dfs

y_dfs = []

for station_ID in station_IDs:
    y_df = pd.read_csv(os.path.join(data_folder, station_dataset_dict[station_ID], "y", station_ID + ".csv"))
    
    y_dfs.append(y_df)

#%% load predictions

model_name = model.method_name



pred_df_dict = {}
for dataset_name in all_dataset_names:
    base_predictions_path = os.path.join(predictions_folder, dataset_name)
    predictions_path = os.path.join(base_predictions_path, preprocessing_hash, model_name, hyperparameter_hash, str(all_cutoffs)+".pickle")
        
    
    with open(predictions_path, 'rb') as handle:
        all_pred_dfs = pickle.load(handle)
    
    temp_dict = {ID.replace(".csv",""):df for ID, df in zip(station_ID_dict[dataset_name], all_pred_dfs)}
    pred_df_dict.update(temp_dict)
    
    
y_pred_dfs = []


# scores_df_dict = {}
# for dataset_name in all_dataset_names:
#     base_scores_path = os.path.join(score_folder, station_dataset_dict[station_ID])
#     scores_path = os.path.join(base_scores_path, preprocessing_hash, model.score_calculation_method_name, hyperparameter_hash,"scores.pickle")
        
    
#     with open(scores_path, 'rb') as handle:
#         all_scores_dfs = pickle.load(handle)
    
#     temp_dict = {ID.replace(".csv",""):df for ID, df in zip(station_ID_dict[dataset_name], all_scores_dfs)}
#     scores_df_dict.update(temp_dict)
    
    
y_pred_dfs = []

for station_ID in station_IDs:
    
    y_pred_dfs.append(pred_df_dict[station_ID])
 
#%% plot the predictions

plot_predictions(X_dfs, y_dfs, y_pred_dfs, station_IDs, model, show_IF_scores=True, show_TP_FP_FN=True, opacity_TP=0.6, pretty_plot=True, which_stations = range(0, len(station_IDs)))