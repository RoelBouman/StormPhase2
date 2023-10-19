#%% Load packages
import os

import numpy as np
from sklearn.model_selection import ParameterGrid
#import pandas as pd

from src.methods import SingleThresholdStatisticalProfiling
from src.methods import DoubleThresholdStatisticalProfiling
from src.methods import SingleThresholdIsolationForest

from src.methods import SingleThresholdBinarySegmentation
from src.methods import DoubleThresholdBinarySegmentation

from src.methods import StackEnsemble
from src.methods import NaiveStackEnsemble

from src.preprocess import preprocess_per_batch_and_write
from src.io_functions import save_dataframe_list, save_model, save_metric, save_PRFAUC_table, save_minmax_stats
from src.io_functions import load_batch, load_model, load_metric, load_PRFAUC_table, load_minmax_stats
from src.io_functions import print_count_nan
from src.evaluation import f_beta, cutoff_averaged_f_beta, calculate_unsigned_absolute_and_relative_stats, calculate_PRFAUC_table

from src.reporting_functions import print_metrics_and_stats
#%% set process variables

data_folder = "data"
result_folder = "results"
intermediates_folder = "intermediates"
model_folder = "saved_models"

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")
metric_folder = os.path.join(result_folder, "metrics")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

beta = 1.5
def score_function(precision, recall):
    return f_beta(precision, recall, beta)

report_metrics_and_stats = True

remove_missing = True

write_csv_intermediates = True

preprocessing_overwrite = False #if set to True, overwrite previous preprocessed data

training_overwrite = True #if
testing_overwrite = True
validation_overwrite = True

#%% define hyperparemeters for preprocessing

preprocessing_hyperparameters = {'subsequent_nr': 5, 'lin_fit_quantiles': (10, 90)}

#%% define hyperparameters per method:

SingleThresholdSP_hyperparameters = {"quantiles":[(5,95), (10,90), (15, 85), (20,80), (25,75)]}

DoubleThresholdSP_hyperparameters = {"quantiles":[(5,95), (10,90), (15, 85), (20,80), (25,75)]}

SingleThresholdIF_hyperparameters = {"n_estimators": [1000]}

SingleThresholdBS_hyperparameters = {"beta": [0.005, 0.008, 0.12, 0.015], "model": ['l1'], 'min_size': [100], "jump": [10], "quantiles": [(5,95)], "scaling": [True], "penalty": ['fused_lasso']}

DoubleThresholdBS_hyperparameters = {"beta": [0.005, 0.008, 0.12, 0.015], "model": ['l1'], 'min_size': [100], "jump": [10], "quantiles": [(5,95)], "scaling": [True], "penalty": ['fused_lasso']}

NaiveStackEnsemble_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, SingleThresholdStatisticalProfiling]], "method_hyperparameter_dict_list":[[{'beta':0.12, 'model':'l1','min_size':100, 'jump':10, 'quantiles':(5,95), 'scaling':True, 'penalty':'fused_lasso'},{'quantiles': (5, 95)}]], "all_cutoffs":[all_cutoffs]}
#%% load Train data
which_split = "Train"

print("Split: Train")
X_train_dfs, y_train_dfs, X_train_files = load_batch(data_folder, which_split)
    
#%% Preprocess Train data
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_type = "basic"
train_file_names = X_train_files

X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, event_lengths_train= preprocess_per_batch_and_write(X_train_dfs, y_train_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, train_file_names, all_cutoffs, preprocessing_hyperparameters, remove_missing)

#%% Detect anomalies/switch events
# Save results, make saving of scores optional, as writing this many results is fairly costly

#%% Training

#methods = {"SingleThresholdSP":SingleThresholdStatisticalProfiling, "DoubleThresholdSP": DoubleThresholdStatisticalProfiling,
#           "SingleThresholdIF":SingleThresholdIsolationForest}
#hyperparameter_dict = {"SingleThresholdSP":SingleThresholdSP_hyperparameters, "DoubleThresholdSP":DoubleThresholdSP_hyperparameters,
#                       "SingleThresholdIF":SingleThresholdIF_hyperparameters}

#methods = {"SingleThresholdIF":SingleThresholdIsolationForest}
#hyperparameter_dict = {"SingleThresholdIF":SingleThresholdIF_hyperparameters}

#DoubleThresholdSP_hyperparameters = {"quantiles":[(5,95)], "used_cutoffs":[all_cutoffs]}
#SingleThresholdSP_hyperparameters = {"quantiles":[(5,95)], "used_cutoffs":[all_cutoffs]}
#methods = {"SingleThresholdSP":SingleThresholdStatisticalProfiling, "DoubleThresholdSP":DoubleThresholdStatisticalProfiling}
#hyperparameter_dict = {"SingleThresholdSP":SingleThresholdSP_hyperparameters, "DoubleThresholdSP":DoubleThresholdSP_hyperparameters}

methods = {"NaiveStackEnsemble":NaiveStackEnsemble}
hyperparameter_dict = {"NaiveStackEnsemble":NaiveStackEnsemble_hyperparameters}


for method_name in methods:
    print("Now training: " + method_name)
    all_hyperparameters = hyperparameter_dict[method_name]
    hyperparameter_list = list(ParameterGrid(all_hyperparameters))
    
    for hyperparameters in hyperparameter_list:
        hyperparameter_string = str(hyperparameters)
        hyperparameter_string = hyperparameter_string.replace(':', '')
        print(hyperparameter_string)
        
        scores_path = os.path.join(score_folder, which_split, method_name, hyperparameter_string)
        predictions_path = os.path.join(predictions_folder, which_split, method_name, hyperparameter_string)
        fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, method_name)
        PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, method_name)
        minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, method_name)
        model_path = os.path.join(model_folder, method_name)
        
        full_model_path = os.path.join(model_path, hyperparameter_string+".pickle")
        if training_overwrite or not os.path.exists(full_model_path):
            
            model = methods[method_name](**hyperparameters, score_function=score_function)
            
            y_train_scores_dfs, y_train_predictions_dfs = model.fit_transform_predict(X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train)
            
            metric = cutoff_averaged_f_beta(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
            
            minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_train_dfs_preprocessed, y_train_dfs_preprocessed, y_train_predictions_dfs, load_column="S_original")
            absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
            PRFAUC_table = calculate_PRFAUC_table(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
            
            save_dataframe_list(y_train_scores_dfs, X_train_files, scores_path, overwrite=training_overwrite)
            save_dataframe_list(y_train_predictions_dfs, X_train_files, predictions_path, overwrite=training_overwrite)
            
            save_metric(metric, fscore_path, hyperparameter_string)
            save_PRFAUC_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_string)
            save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_string)
            
            save_model(model, model_path, hyperparameter_string)
        else:
            print("Model already evaluated, loading results instead:")
            metric = load_metric(fscore_path, hyperparameter_string)
            PRFAUC_table = load_PRFAUC_table(PRFAUC_table_path, hyperparameter_string)
            minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_string)
            
            model = load_model(model_path, hyperparameter_string)
            
        if model.is_single_threshold_method: 
            
            print("Optimal threshold:" )
            print(model.optimal_threshold)
        else: 
            print("Optimal thresholds:")
            print((model.optimal_negative_threshold, model.optimal_positive_threshold))
            
        if report_metrics_and_stats:
            print_metrics_and_stats(metric, minmax_stats, PRFAUC_table)

#%% Test
#%% load Test data
which_split = "Test"
print("Split: Test")
X_test_dfs, y_test_dfs, X_test_files = load_batch(data_folder, which_split)
    
#%% Preprocess Test data
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_type = "basic"
test_file_names = X_test_files

X_test_dfs_preprocessed, y_test_dfs_preprocessed, label_filters_for_all_cutoffs_test, event_lengths_test = preprocess_per_batch_and_write(X_test_dfs, y_test_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, test_file_names, all_cutoffs, preprocessing_hyperparameters, remove_missing)

#%% run Test evaluation:
    
best_hyperparameters = {}

for method_name in methods:
    print("Now testing: " + method_name)
    all_hyperparameters = hyperparameter_dict[method_name]
    hyperparameter_list = list(ParameterGrid(all_hyperparameters))
    
    highest_test_metric = -np.inf
    
    for hyperparameters in hyperparameter_list:
        hyperparameter_string = str(hyperparameters)
        hyperparameter_string = hyperparameter_string.replace(':', '')
        print(hyperparameter_string)
        
        scores_path = os.path.join(score_folder, which_split, method_name, hyperparameter_string)
        predictions_path = os.path.join(predictions_folder, which_split, method_name, hyperparameter_string)
        fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, method_name)
        PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, method_name)
        minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, method_name)
        model_path = os.path.join(model_folder, method_name)
        
        full_metric_path = os.path.join(fscore_path, hyperparameter_string+".csv")
        if testing_overwrite or not os.path.exists(full_metric_path):
            
            model = load_model(model_path, hyperparameter_string)
            
            y_test_scores_dfs, y_test_predictions_dfs = model.transform_predict(X_test_dfs_preprocessed, y_test_dfs_preprocessed, label_filters_for_all_cutoffs_test)
            
            test_metric = cutoff_averaged_f_beta(y_test_dfs_preprocessed, y_test_predictions_dfs, label_filters_for_all_cutoffs_test, beta)
            
            minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_test_dfs_preprocessed, y_test_dfs_preprocessed, y_test_predictions_dfs, load_column="S_original")
            absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
            PRFAUC_table = calculate_PRFAUC_table(y_test_dfs_preprocessed, y_test_predictions_dfs, label_filters_for_all_cutoffs_test, beta)

            save_dataframe_list(y_test_scores_dfs, X_test_files, scores_path, overwrite=testing_overwrite)
            save_dataframe_list(y_test_predictions_dfs, X_test_files, predictions_path, overwrite=testing_overwrite)
            
            save_metric(test_metric, fscore_path, hyperparameter_string)
            save_PRFAUC_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_string)
            save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_string)
        else:
            print("Model already evaluated, loading results instead:")
            test_metric = load_metric(fscore_path, hyperparameter_string)
            PRFAUC_table = load_PRFAUC_table(PRFAUC_table_path, hyperparameter_string)
            minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_string)
            
        if test_metric > highest_test_metric:
            highest_test_metric = test_metric
            best_hyperparameters[method_name] = hyperparameters
            
        if report_metrics_and_stats:
            print_metrics_and_stats(test_metric, minmax_stats, PRFAUC_table)
    

#%% Validation
#%% load Validation data
which_split = "Validation"
print("Split: Validation")
X_val_dfs, y_val_dfs, X_val_files = load_batch(data_folder, which_split)
    
#%% Preprocess Validation data
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_type = "basic"
val_file_names = X_val_files

X_val_dfs_preprocessed, y_val_dfs_preprocessed, label_filters_for_all_cutoffs_val, event_lengths_val, = preprocess_per_batch_and_write(X_val_dfs, y_val_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, val_file_names, all_cutoffs, preprocessing_hyperparameters, remove_missing)

#%% run Validation evaluation:


for method_name in methods:
    print("Now validating: " + method_name)
    
    hyperparameters = best_hyperparameters[method_name]
    
    hyperparameter_string = str(hyperparameters)
    hyperparameter_string = hyperparameter_string.replace(':', '')
    print(hyperparameter_string)
    
    scores_path = os.path.join(score_folder, which_split, method_name, hyperparameter_string)
    predictions_path = os.path.join(predictions_folder, which_split, method_name, hyperparameter_string)
    fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, method_name)
    PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, method_name)
    minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, method_name)
    model_path = os.path.join(model_folder, method_name)
    
    full_metric_path = os.path.join(fscore_path, hyperparameter_string+".csv")
    if validation_overwrite or not os.path.exists(full_metric_path):
        
        model = load_model(model_path, hyperparameter_string)
        
        y_val_scores_dfs, y_val_predictions_dfs = model.transform_predict(X_val_dfs_preprocessed, y_val_dfs_preprocessed, label_filters_for_all_cutoffs_val)
        
        val_metric = cutoff_averaged_f_beta(y_val_dfs_preprocessed, y_val_predictions_dfs, label_filters_for_all_cutoffs_val, beta)
        
        minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_val_dfs_preprocessed, y_val_dfs_preprocessed, y_val_predictions_dfs, load_column="S_original")
        absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
        PRFAUC_table = calculate_PRFAUC_table(y_val_dfs_preprocessed, y_val_predictions_dfs, label_filters_for_all_cutoffs_val, beta)
        
        save_dataframe_list(y_val_scores_dfs, X_val_files, scores_path, overwrite=validation_overwrite)
        save_dataframe_list(y_val_predictions_dfs, X_val_files, predictions_path, overwrite=validation_overwrite)
        
        save_metric(val_metric, fscore_path, hyperparameter_string)
        save_PRFAUC_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_string)
        save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_string)
    else:
        print("Model already evaluated, loading results instead:")
        val_metric = load_metric(fscore_path, hyperparameter_string)
        PRFAUC_table = load_PRFAUC_table(PRFAUC_table_path, hyperparameter_string)
        minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_string)
            
    if report_metrics_and_stats:
        print_metrics_and_stats(val_metric, minmax_stats, PRFAUC_table)
    