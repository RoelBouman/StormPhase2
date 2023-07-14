#%% Load packages
import os

import numpy as np
from sklearn.model_selection import ParameterGrid
#import pandas as pd

from src.methods import SingleThresholdStatisticalProfiling
from src.preprocess import preprocess_per_batch_and_write
from src.io_functions import save_dataframe_list, save_model, save_metric
from src.io_functions import load_batch, load_model, load_metric
from src.evaluation import f_beta, cutoff_averaged_f_beta

#%% set process variables

data_folder = "data"
result_folder = "results"
intermediates_folder = "intermediates"
model_folder = "saved_models"

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")
metric_folder = os.path.join(result_folder, "metrics")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

beta = 10
def score_function(precision, recall):
    return f_beta(precision, recall, beta)

remove_missing = True

write_csv_intermediates = True

preprocessing_overwrite = False #if set to True, overwrite previous preprocessed data

training_overwrite = False

#%% define hyperparameters per method:
SingleThresholdSP_hyperparameters = {"quantiles":[(5,95), (10,90), (15, 85), (20,80), (25,75)]}
#%% load Train data
# Do not load data if preprocessed data is available already
which_split = "Train"

print("Split: Train")
X_train_dfs, y_train_dfs, X_train_files = load_batch(data_folder, which_split)
    
#%% Preprocess Train data
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_type = "basic"
train_file_names = X_train_files

X_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, event_lengths_train= preprocess_per_batch_and_write(X_train_dfs, y_train_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, train_file_names, all_cutoffs, remove_missing)


#%% Detect anomalies/switch events
# Save results, make saving of scores optional, as writing this many results is fairly costly

#%% Training

methods = {"SingleThresholdSP":SingleThresholdStatisticalProfiling}
hyperparameter_dict = {"SingleThresholdSP":SingleThresholdSP_hyperparameters}

best_hyperparameters = {}

for method_name in methods:
    print("Now training: " + method_name)
    all_hyperparameters = hyperparameter_dict[method_name]
    hyperparameter_list = list(ParameterGrid(all_hyperparameters))
    
    highest_train_metric = -np.inf
    
    for hyperparameters in hyperparameter_list:
        hyperparameter_string = str(hyperparameters)
        print(hyperparameter_string)
        
        scores_path = os.path.join(score_folder, which_split, method_name, hyperparameter_string)
        predictions_path = os.path.join(predictions_folder, which_split, method_name, hyperparameter_string)
        metric_path = os.path.join(metric_folder, which_split, method_name)
        model_path = os.path.join(model_folder, method_name)
        
        full_model_path = os.path.join(model_path, hyperparameter_string+".pickle")
        if training_overwrite or not os.path.exists(full_model_path):
            
            model = methods[method_name](**hyperparameters, score_function=score_function)
            
            y_train_scores_dfs, y_train_predictions_dfs = model.fit_transform_predict(X_train_dfs_preprocessed, y_train_dfs, label_filters_for_all_cutoffs_train)
            optimal_threshold = model.optimal_threshold_
            
            train_metric = cutoff_averaged_f_beta(y_train_dfs, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
            
            save_dataframe_list(y_train_scores_dfs, X_train_files, scores_path, overwrite=training_overwrite)
            save_dataframe_list(y_train_predictions_dfs, X_train_files, predictions_path, overwrite=training_overwrite)
            
            save_metric(train_metric, metric_path, hyperparameter_string, overwrite=training_overwrite)
            save_model(model, model_path, hyperparameter_string, overwrite=training_overwrite)
        else:
            train_metric = load_metric(metric_path, hyperparameter_string)
            model = load_model(model_path, hyperparameter_string)
            

            
        print("Optimal threshold:" )
        print(model.optimal_threshold_)
        print("Train metric:" )
        print(train_metric)



#%% Test
#%% load Test data
# Do not load data if preprocessed data is available already
which_split = "Test"
print("Split: Test")
X_test_dfs, y_test_dfs, X_test_files = load_batch(data_folder, which_split)
    
#%% Preprocess Test data
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_type = "basic"
test_file_names = X_test_files

X_test_dfs_preprocessed, labels_for_all_cutoffs_test, event_lengths_test = preprocess_per_batch_and_write(X_test_dfs, y_test_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, test_file_names, all_cutoffs, remove_missing)

#%% run Test evaluation:
    
        # if train_metric > highest_train_metric:
        #     highest_train_metric = train_metric
        #     best_hyperparameters = hyperparameters
#%% Validation
#%% load Validation data
# Do not load data if preprocessed data is available already
which_split = "Validation"
print("Split: Validation")
X_val_dfs, y_val_dfs, X_val_files = load_batch(data_folder, which_split)
    
#%% Preprocess Validation data
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_type = "basic"
val_file_names = X_val_files

X_val_dfs_preprocessed, labels_for_all_cutoffs_val, event_lengths_val, = preprocess_per_batch_and_write(X_val_dfs, y_val_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, val_file_names, all_cutoffs, remove_missing)
