#%% package loading
import os
import pickle
import jsonpickle
import sqlite3

import pandas as pd
import numpy as np

from hashlib import sha256

import seaborn as sns

import matplotlib.pyplot as plt

from src.plot_functions import plot_single_prediction

from src.methods import SingleThresholdStatisticalProcessControl
from src.methods import DoubleThresholdStatisticalProcessControl

from src.methods import SingleThresholdIsolationForest

from src.methods import SingleThresholdBinarySegmentation
from src.methods import DoubleThresholdBinarySegmentation

# from src.methods import StackEnsemble
# from src.methods import NaiveStackEnsemble
from src.methods import SequentialEnsemble


from src.preprocess import preprocess_per_batch_and_write

from src.evaluation import f_beta

sns.set()

#%% plot options:
    
plt.rcParams['axes.labelsize'] = 40  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 35  # Font size for x tick labels
plt.rcParams['ytick.labelsize'] = 35  # Font size for y tick labels
plt.rcParams['legend.fontsize'] = 32  # Font size for legend
#%% Define user parameters:
# choose station IDs per method:
station_IDs_per_method = {#"DoubleThresholdBS":['8.csv'], #'042.csv', '089.csv', '17.csv', '96.csv'
                          #"SingleThresholdSPC":["17.csv"],
                          "Sequential-SingleThresholdBS+SingleThresholdSPC":["042.csv"],
                           }
#station_IDs_per_method = {"Sequential-SingleThresholdBS+SingleThresholdSPC":[str(ID)+".csv" for ID in range(1,202) if ID not in [8, 25, 35, 70, 106, 130, 190]]}

                          # }

save_predictions = True
best_predictions_folder = "best_route_labels"
os.makedirs(best_predictions_folder, exist_ok=True)

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

preprocessed_X_folder = os.path.join(intermediates_folder, "preprocessed_data_csvs")
label_filter_folder = os.path.join(intermediates_folder, "label_filters_per_cutoff_csvs")

train_name = "Train"
test_name = "Test"
validation_name = "Validation"

all_dataset_names = [train_name, test_name, validation_name]

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

#%% connect to database

DBFILE = dataset+"_experiment_results.db" #TODO: remove copy
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()

#%% Set other plotting parameters

beta = 1.5

model_dict = {  "SingleThresholdIF":SingleThresholdIsolationForest,
                "SingleThresholdBS":SingleThresholdBinarySegmentation, 
                "SingleThresholdSPC":SingleThresholdStatisticalProcessControl,
                
                "DoubleThresholdBS":DoubleThresholdBinarySegmentation, 
                "DoubleThresholdSPC":DoubleThresholdStatisticalProcessControl, 
                
                
                "Sequential-SingleThresholdBS+SingleThresholdSPC":SequentialEnsemble, 
                "Sequential-DoubleThresholdBS+DoubleThresholdSPC":SequentialEnsemble,
                "Sequential-SingleThresholdBS+DoubleThresholdSPC":SequentialEnsemble,
                "Sequential-DoubleThresholdBS+SingleThresholdSPC":SequentialEnsemble,
                
                
                "Sequential-SingleThresholdBS+SingleThresholdIF":SequentialEnsemble, 
                "Sequential-DoubleThresholdBS+SingleThresholdIF":SequentialEnsemble,
            }
#%% Run plotting procedure


for method_name in station_IDs_per_method:
    plot_station_IDs = station_IDs_per_method[method_name]
    
    station_IDs = [station_ID.replace(".csv", "") for station_ID in plot_station_IDs]
    #station_IDs = ["1","041"]
    
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
    
    #%% hyperparameter selection
    
    # use your own hyperparameters
    # use best model hyperparameters
    
    # best_model_entry = db_cursor.execute("SELECT e.* FROM experiment_results e WHERE e.metric = (SELECT MAX(metric)FROM experiment_results WHERE method = (?) AND which_split = (?))", (method_name, "Validation"))
    # (preprocessing_hash, hyperparameter_hash, _, _, _, _, _) = next(best_model_entry)
    
    # db_result = db_cursor.execute("SELECT method_hyperparameters FROM experiment_results WHERE preprocessing_hash='{}' AND hyperparameter_hash='{}'".format(preprocessing_hash, hyperparameter_hash)).fetchone()[0]
    # model_hyperparameters = jsonpickle.loads(db_result)
    
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

    model_hyperparameters = jsonpickle.decode(hyperparameter_string_pickle, keys=True)
    preprocessing_hyperparameters = jsonpickle.decode(preprocessing_hyperparameter_string_pickle, keys=True)

    
    #%% load model
    
    model = model_dict[method_name](model_folder, preprocessing_hash, **model_hyperparameters)
    #model = SingleThresholdStatisticalProcessControl(model_folder, preprocessing_hash, **model_hyperparameters, score_function=score_function)
    #model = SingleThresholdIsolationForest(model_folder, preprocessing_hash, **model_hyperparameters, score_function=score_function)
    # model = SingleThresholdBinarySegmentation(model_folder, preprocessing_hash, **model_hyperparameters, score_function=score_function)
    # model = DoubleThresholdBinarySegmentation(model_folder, preprocessing_hash, **model_hyperparameters, score_function=score_function)
    
    # get hash (if not using best model) for prediction loading
    hyperparameter_hash = model.get_hyperparameter_hash()
    
    #%% load preprocessed X dfs
    
    X_dfs = []
    
    for station_ID in station_IDs:
        X_df = pd.read_csv(os.path.join(preprocessed_X_folder, station_dataset_dict[station_ID], preprocessing_hash, "X", station_ID + ".csv"))
        
        X_dfs.append(X_df)
    
    #%% load preprocessed y dfs
    
    y_dfs = []
    
    for station_ID in station_IDs:
        y_df = pd.read_csv(os.path.join(preprocessed_X_folder, station_dataset_dict[station_ID], preprocessing_hash, "y", station_ID + ".csv"))
        
        y_dfs.append(y_df)
        
    #%% load label_filters
    
    label_filters_for_all_cutoffs = []
    
    for station_ID in station_IDs:
        label_filter_for_all_cutoffs = pd.read_csv(os.path.join(label_filter_folder, station_dataset_dict[station_ID], preprocessing_hash, station_ID + ".csv"))
        
        label_filters_for_all_cutoffs.append(label_filter_for_all_cutoffs)
    
    #%% load predictions
    
    model_name = model.method_name
    
    
    
    pred_df_dict = {}
    scores_df_dict = {}
    try:
        for dataset_name in all_dataset_names:
            base_predictions_path = os.path.join(predictions_folder, dataset_name)
            predictions_path = os.path.join(base_predictions_path, preprocessing_hash, model_name, hyperparameter_hash, str(all_cutoffs)+".pickle")
        
            with open(predictions_path, 'rb') as handle:
                all_pred_dfs = pickle.load(handle)
            temp_dict = {ID.replace(".csv",""):df for ID, df in zip(station_ID_dict[dataset_name], all_pred_dfs)}
    
            pred_df_dict.update(temp_dict)
        
    except FileNotFoundError:
        print("Results can't be reloaded, recalculating explicitly:")
        # base_scores_path = os.path.join(score_folder, dataset_name)
        # base_predictions_path = os.path.join(predictions_folder, dataset_name)
        # base_intermediates_path = os.path.join(intermediates_folder, dataset_name)
        
        # scores_path = os.path.join(base_scores_path, preprocessing_hash)
        # predictions_path = os.path.join(base_predictions_path, preprocessing_hash)
        # intermediates_path = os.path.join(base_intermediates_path, preprocessing_hash)
        
        
        scores_path = "temp"
        predictions_path = "temp"
        intermediates_path = "temp"
        
        #X_dfs_preprocessed, y_dfs_preprocessed, label_filters_for_all_cutoffs, event_lengths = preprocess_per_batch_and_write(X_dfs, y_dfs, intermediates_folder, dataset_name, False, False, file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, True, False)

        _, all_pred_dfs = model.transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path=scores_path, base_predictions_path=predictions_path, base_intermediates_path=intermediates_path, overwrite=False, verbose=False, dry_run=True)
        temp_dict = {ID.replace(".csv",""):df for ID, df in zip(station_IDs, all_pred_dfs)}
        pred_df_dict.update(temp_dict)
        
        
    y_pred_dfs = []
    for station_ID in station_IDs:
        
        y_pred_dfs.append(pred_df_dict[station_ID])
     
    
    #%% plot the predictions
    show_IF_scores=False
    show_TP_FP_FN=True
    opacity_TP=0.6
    pretty_plot=True
    
    which_stations = range(0, len(station_IDs))
    n_stations_if_random = 3
    # select random stations if no stations selected
    if which_stations == None:
        which_stations = np.random.randint(0, len(X_dfs), n_stations_if_random)
    
    for station in which_stations:
        X_df = X_dfs[station]
        y_df = y_dfs[station]
        y_pred_df = y_pred_dfs[station]
        file = station_IDs[station]
        
        plot_single_prediction(X_df, y_df, y_pred_df, file, model, show_IF_scores = show_IF_scores, show_TP_FP_FN = show_TP_FP_FN, opacity_TP = opacity_TP, pretty_plot = pretty_plot)
        
        base_plot_path = os.path.join(figure_folder, "prediction_plots", method_name)
        os.makedirs(base_plot_path, exist_ok=True)
        plt.savefig(os.path.join(base_plot_path, station_IDs[station]  + ".png"), format="png")
        plt.savefig(os.path.join(base_plot_path, station_IDs[station]  + ".pdf"), format="pdf")
        
        plt.show()
        
        if save_predictions:
            csv_name = os.path.join(best_predictions_folder, str(station)+".csv")
            y_pred_df["M_TIMESTAMP"] = X_df["M_TIMESTAMP"]
            
            y_pred_df.to_csv(csv_name)
