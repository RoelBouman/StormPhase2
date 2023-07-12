#%% Load packages
import os

import numpy as np
#import pandas as pd

from src.methods import SingleThresholdStatisticalProfiling
from src.preprocess import preprocess_per_batch_and_write
from src.io_functions import save_dataframe_list, load_batch
from src.evaluation import f_beta
#%% set process variables

data_folder = "data"
result_folder = "results"
intermediates_folder = "intermediates"

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

beta = 10
score_function = lambda precision, recall: f_beta(precision, recall, beta)

remove_missing = True

write_csv_intermediates = True

preprocessing_overwrite = False #if set to True, overwrite previous preprocessed data


#%% define evaluation functions:
    


        
#%% load Train data
# Do not load data if preprocessed data is available already
which_split = "Train"

print("Split: Train")
X_train_dfs, y_train_dfs, X_train_files = load_batch(data_folder, which_split)
    
#%% Preprocess Train data
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_type = "basic"
file_names = X_train_files

X_train_dfs_preprocessed, label_filters_for_all_cutoffs, event_lengths, preprocessing_parameters = preprocess_per_batch_and_write(X_train_dfs, y_train_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, file_names, all_cutoffs, remove_missing)


#%% Detect anomalies/switch events
# Save results, make saving of scores optional, as writing this many results is fairly costly

#%% Train evaluation

#Define hyperparameter range:
    
#hyperparameter_dict = {"quantiles":[(10,90), (20,80), (25,75), (5,95)]}



method = SingleThresholdStatisticalProfiling(score_function=score_function)
method_name = "SingleThresholdSP"
#save best scores:
#TODO: define method for definitive selection


y_train_scores_dfs, y_train_predictions_dfs = method.fit_transform_predict(X_train_dfs_preprocessed, y_train_dfs, label_filters_for_all_cutoffs)
optimal_threshold = method.optimal_threshold_

scores_path = os.path.join(score_folder, method_name, str(optimal_threshold), which_split)
predictions_path = os.path.join(predictions_folder, method_name, str(optimal_threshold), which_split)

save_dataframe_list(y_train_scores_dfs, X_train_files, scores_path, overwrite=False)
save_dataframe_list(y_train_predictions_dfs, X_train_files, predictions_path, overwrite=False)

#train_result_df = SP.train_result_df_
#best_model = SP.best_model_
#best_hyperparameters = SP.best_hyperparameters_


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
file_names = X_test_files

X_test_dfs_preprocessed, labels_for_all_cutoffs, event_lengths, preprocessing_parameters = preprocess_per_batch_and_write(X_test_dfs, y_test_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, file_names, all_cutoffs, remove_missing)

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
file_names = X_val_files

X_val_dfs_preprocessed, labels_for_all_cutoffs, event_lengths, preprocessing_parameters = preprocess_per_batch_and_write(X_val_dfs, y_val_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, file_names, all_cutoffs, remove_missing)
