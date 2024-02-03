#%% package loading
import os
import pickle
import jsonpickle
import sqlite3

import pandas as pd
import numpy as np

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

preprocessed_folder = os.path.join(intermediates_folder, "preprocessed_data_csvs")

train_name = "Train"
test_name = "Test"
validation_name = "Validation"

all_dataset_names = [train_name, test_name, validation_name]

#%%

DBFILE = "experiment_results.db"
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()

#%% choose station IDs

# IDs must be from same split
station_IDs = ["1", "041", "019"]

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
    
#%% load model

#load model:
preprocessing_hash = "10cab9fc324db7a2fd5d8674c71edb68908b5e572ffa442d201eb0ca0aa288e1"
hyperparameter_hash = "863c7a1a49f110ada1d11bf21549b9f60f53c72042a80a36a0969583a18d42e1"

db_result = db_cursor.execute("SELECT method_hyperparameters FROM experiment_results WHERE preprocessing_hash='{}' AND hyperparameter_hash='{}'".format(preprocessing_hash, hyperparameter_hash)).fetchone()[0]

hyperparameters = jsonpickle.loads(db_result)

beta = 1.5
def score_function(precision, recall):
    return f_beta(precision, recall, beta)

model = SingleThresholdStatisticalProcessControl(model_folder, preprocessing_hash, **hyperparameters, score_function=score_function)

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

base_predictions_path = os.path.join(predictions_folder, station_dataset_dict[station_ID])
predictions_path = os.path.join(base_predictions_path, model_name, preprocessing_hash, hyperparameter_hash)
    
y_pred_dfs = []

for station_ID in station_IDs:
    y_df = pd.read_csv(os.path.join(predictions_path, "stations", station_ID + ".csv"))
    
    y_pred_dfs.append(y_df)
 
#%% plot the predictions

plot_predictions(X_dfs, y_dfs, y_pred_dfs, station_IDs, model, show_TP_FP_FN=True, opacity_TP=0.6, pretty_plot=True, which_stations = range(0, len(station_IDs)))