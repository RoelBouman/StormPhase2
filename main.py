#%% Load packages
import os
import sqlite3
import jsonpickle

import numpy as np
from sklearn.model_selection import ParameterGrid
#import pandas as pd
from hashlib import sha256

from src.methods import SingleThresholdStatisticalProcessControl
from src.methods import DoubleThresholdStatisticalProcessControl


from src.methods import SingleThresholdIsolationForest

from src.methods import SingleThresholdBinarySegmentation
from src.methods import DoubleThresholdBinarySegmentation

from src.methods import StackEnsemble
from src.methods import NaiveStackEnsemble
from src.methods import SequentialEnsemble

from src.preprocess import preprocess_per_batch_and_write
from src.io_functions import save_metric, save_table, save_minmax_stats
from src.io_functions import load_batch, load_metric, load_table, load_minmax_stats
from src.evaluation import cutoff_averaged_f_beta, calculate_unsigned_absolute_and_relative_stats, calculate_PRFAUC_table, calculate_bootstrap_stats


from src.reporting_functions import print_metrics_and_stats, bootstrap_stats_to_printable

#%% set process variables

dataset = "OS_data" #alternatively: route_data
data_folder = os.path.join("data", dataset)
result_folder = os.path.join("results", dataset)
intermediates_folder = os.path.join("intermediates", dataset)
model_folder = os.path.join("saved_models", dataset)

score_folder = os.path.join(result_folder, "scores")
predictions_folder = os.path.join(result_folder, "predictions")
metric_folder = os.path.join(result_folder, "metrics")

all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

beta = 1.5
report_metrics_and_stats = True

remove_missing = True

write_csv_intermediates = True

preprocessing_overwrite = False #if set to True, overwrite previous preprocessed data

training_overwrite = True 
validation_overwrite = False
testing_overwrite = False

bootstrap_validation = True
bootstrap_iterations = 10000

dry_run = False

verbose = False

model_test_run = True #Only run 1 hyperparameter setting per model type if True

#%% set up database

DBFILE = dataset+"_experiment_results.db"
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()
if not database_exists:
    db_cursor.execute("CREATE TABLE experiment_results(preprocessing_hash, hyperparameter_hash, method, which_split, preprocessing_hyperparameters, method_hyperparameters, metric, PRIMARY KEY (preprocessing_hash, hyperparameter_hash, method, which_split))")
    
#%% define hyperparemeters for preprocessing


if dataset == "OS_data":
    all_preprocessing_hyperparameters = {'subsequent_nr': [5], 'lin_fit_quantiles': [(10, 90)], "label_transform_dict": [{0:0, 1:1, 4:5, 5:5}], "remove_uncertain": [False]}
elif dataset == "route_data":
    all_preprocessing_hyperparameters = {'subsequent_nr': [5], 'lin_fit_quantiles': [(10, 90)], "label_transform_dict": [{0:0, 1:1, 4:5, 5:5}], "remove_uncertain": [True, False]}

#%% define hyperparameters per method:

#For IF, pass sequence of dicts to avoid useless hyperparam combos (such as scaling=True, forest_per_station=True)
SingleThresholdIF_hyperparameters_no_scaling = [{"n_estimators": [1000], 
                                                 "forest_per_station":[True], 
                                                 "scaling":[False], 
                                                 "score_function_kwargs":[{"beta":beta}]}]

SingleThresholdIF_hyperparameters_scaling = [{"n_estimators": [10000], 
                                              "forest_per_station":[False], 
                                              "scaling":[True], 
                                              "quantiles":[(10,90), (15, 85), (20,80)], 
                                              "score_function_kwargs":[{"beta":beta}]}]

SingleThresholdIF_hyperparameters = SingleThresholdIF_hyperparameters_no_scaling + SingleThresholdIF_hyperparameters_scaling

SingleThresholdSPC_hyperparameters = {"quantiles":[(10,90), (15, 85), (20,80)], 
                                      "score_function_kwargs":[{"beta":beta}]}

DoubleThresholdSPC_hyperparameters = SingleThresholdSPC_hyperparameters

SingleThresholdBS_hyperparameters = {"beta": [0.005, 0.008, 0.015, 0.05, 0.08, 0.12], 
                                     "model": ['l1'], 
                                     'min_size': [150, 200, 288], 
                                     "jump": [5, 10], 
                                     "quantiles": [(10,90), (15, 85), (20,80)], 
                                     "scaling": [True], 
                                     "penalty": ['L1'], 
                                     "reference_point":["mean", "median", "longest_median", "longest_mean"], 
                                     "score_function_kwargs":[{"beta":beta}]}

DoubleThresholdBS_hyperparameters = SingleThresholdBS_hyperparameters


ensemble_method_hyperparameter_dict_list = list(ParameterGrid({0:list(ParameterGrid(SingleThresholdBS_hyperparameters)), 
                                                         1:list(ParameterGrid(SingleThresholdSPC_hyperparameters))}))
ensemble_method_hyperparameter_dict_list = [list(l.values()) for l in ensemble_method_hyperparameter_dict_list]


SingleThresholdBS_SingleThresholdSPC_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]], 
                                                        "method_hyperparameter_dict_list":ensemble_method_hyperparameter_dict_list, 
                                                        "cutoffs_per_method":[[all_cutoffs[2:], all_cutoffs[:2]]]}

DoubleThresholdBS_DoubleThresholdSPC_hyperparameters = SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
DoubleThresholdBS_DoubleThresholdSPC_hyperparameters["method_classes"] = [[DoubleThresholdBinarySegmentation, DoubleThresholdStatisticalProcessControl]]

DoubleThresholdBS_SingleThresholdSPC_hyperparameters = SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
DoubleThresholdBS_SingleThresholdSPC_hyperparameters["method_classes"] = [[DoubleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]]

SingleThresholdBS_DoubleThresholdSPC_hyperparameters = SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
SingleThresholdBS_DoubleThresholdSPC_hyperparameters["method_classes"] = [[SingleThresholdBinarySegmentation, DoubleThresholdStatisticalProcessControl]]


Naive_SingleThresholdBS_SingleThresholdSPC_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]], 
                                                              "method_hyperparameter_dict_list":ensemble_method_hyperparameter_dict_list, 
                                                              "all_cutoffs":[all_cutoffs]}

Naive_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters = Naive_SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
Naive_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters["method_classes"] = [[DoubleThresholdBinarySegmentation, DoubleThresholdStatisticalProcessControl]]

Naive_DoubleThresholdBS_SingleThresholdSPC_hyperparameters = Naive_SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
Naive_DoubleThresholdBS_SingleThresholdSPC_hyperparameters["method_classes"] = [[DoubleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]]

Naive_SingleThresholdBS_DoubleThresholdSPC_hyperparameters = Naive_SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
Naive_SingleThresholdBS_DoubleThresholdSPC_hyperparameters["method_classes"] = [[SingleThresholdBinarySegmentation, DoubleThresholdStatisticalProcessControl]]

Sequential_SingleThresholdBS_SingleThresholdSPC_hyperparameters = {"segmentation_method":[SingleThresholdBinarySegmentation], 
                                                                   "anomaly_detection_method":[SingleThresholdStatisticalProcessControl], 
                                                                   "method_hyperparameter_dict_list":[[list(ParameterGrid(SingleThresholdBS_hyperparameters))[2],
                                                                                                       list(ParameterGrid(SingleThresholdSPC_hyperparameters))[0]]], 
                                                                   "cutoffs_per_method":[[all_cutoffs[2:], all_cutoffs[:2]]]}

Sequential_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters = Sequential_SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
Sequential_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters["segmentation_method"] = [DoubleThresholdBinarySegmentation]
Sequential_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters["anomaly_detection_method"] = [DoubleThresholdStatisticalProcessControl]

Sequential_DoubleThresholdBS_SingleThresholdSPC_hyperparameters = Sequential_SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
Sequential_DoubleThresholdBS_SingleThresholdSPC_hyperparameters["segmentation_method"] = [DoubleThresholdBinarySegmentation]

Sequential_SingleThresholdBS_DoubleThresholdSPC_hyperparameters = Sequential_SingleThresholdBS_SingleThresholdSPC_hyperparameters.copy()
Sequential_SingleThresholdBS_DoubleThresholdSPC_hyperparameters["anomaly_detection_method"] = [DoubleThresholdStatisticalProcessControl]
#%% define methods:


methods = { #"SingleThresholdIF":SingleThresholdIsolationForest,
            # "SingleThresholdBS":SingleThresholdBinarySegmentation, 
            # "SingleThresholdSPC":SingleThresholdStatisticalProcessControl,
            
            # "DoubleThresholdBS":DoubleThresholdBinarySegmentation, 
            # "DoubleThresholdSPC":DoubleThresholdStatisticalProcessControl, 
            
            "Naive-SingleThresholdBS+SingleThresholdSPC":NaiveStackEnsemble, 
            # "Naive-DoubleThresholdBS+DoubleThresholdSPC":NaiveStackEnsemble,
            # "Naive-SingleThresholdBS+DoubleThresholdSPC":NaiveStackEnsemble,
            # "Naive-DoubleThresholdBS+SingleThresholdSPC":NaiveStackEnsemble,
            # 
            "SingleThresholdBS+SingleThresholdSPC":StackEnsemble, 
            # "DoubleThresholdBS+DoubleThresholdSPC":StackEnsemble, 
            # "SingleThresholdBS+DoubleThresholdSPC":StackEnsemble,
            # "DoubleThresholdBS+SingleThresholdSPC":StackEnsemble,
            
            # "Sequential-SingleThresholdBS+SingleThresholdSPC":SequentialEnsemble, 
            # "Sequential-DoubleThresholdBS+DoubleThresholdSPC":SequentialEnsemble,
            # "Sequential-SingleThresholdBS+DoubleThresholdSPC":SequentialEnsemble,
            # "Sequential-DoubleThresholdBS+SingleThresholdSPC":SequentialEnsemble
            }

hyperparameter_dict = {"SingleThresholdIF":SingleThresholdIF_hyperparameters,
                       "SingleThresholdSPC":SingleThresholdSPC_hyperparameters, 
                       "SingleThresholdBS":SingleThresholdBS_hyperparameters, 
                       
                       "DoubleThresholdBS":DoubleThresholdBS_hyperparameters, 
                       "DoubleThresholdSPC":DoubleThresholdSPC_hyperparameters, 
                       
                       "Naive-SingleThresholdBS+SingleThresholdSPC":Naive_SingleThresholdBS_SingleThresholdSPC_hyperparameters, 
                       "Naive-DoubleThresholdBS+DoubleThresholdSPC":Naive_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters,
                       "Naive-SingleThresholdBS+DoubleThresholdSPC":Naive_SingleThresholdBS_DoubleThresholdSPC_hyperparameters,
                       "Naive-DoubleThresholdBS+SingleThresholdSPC":Naive_DoubleThresholdBS_SingleThresholdSPC_hyperparameters,
                       
                       "SingleThresholdBS+SingleThresholdSPC":SingleThresholdBS_SingleThresholdSPC_hyperparameters, 
                       "DoubleThresholdBS+DoubleThresholdSPC":DoubleThresholdBS_DoubleThresholdSPC_hyperparameters, 
                       "SingleThresholdBS+DoubleThresholdSPC":SingleThresholdBS_DoubleThresholdSPC_hyperparameters,
                       "DoubleThresholdBS+SingleThresholdSPC":DoubleThresholdBS_SingleThresholdSPC_hyperparameters,

                       "Sequential-SingleThresholdBS+SingleThresholdSPC":Sequential_SingleThresholdBS_SingleThresholdSPC_hyperparameters, 
                       "Sequential-DoubleThresholdBS+DoubleThresholdSPC":Sequential_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters,
                       "Sequential-SingleThresholdBS+DoubleThresholdSPC":Sequential_SingleThresholdBS_DoubleThresholdSPC_hyperparameters,
                       "Sequential-DoubleThresholdBS+SingleThresholdSPC":Sequential_DoubleThresholdBS_SingleThresholdSPC_hyperparameters,
                       }

#%% Preprocess Train data and run algorithms:
# Peprocess entire batch
# Save preprocessed data for later recalculations

# load Train data
which_split = "Train"
print("-----------------------------------------------")
print("Split: Train")
print("-----------------------------------------------")
X_train_dfs, y_train_dfs, X_train_files = load_batch(data_folder, which_split)

train_file_names = X_train_files

base_scores_path = os.path.join(score_folder, which_split)
base_predictions_path = os.path.join(predictions_folder, which_split)
base_intermediates_path = os.path.join(intermediates_folder, which_split)

preprocessing_hyperparameter_list = list(ParameterGrid(all_preprocessing_hyperparameters))
for preprocessing_hyperparameters in preprocessing_hyperparameter_list:
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()
    
    print("-----------------------------------------------")
    print("Now preprocessing: ")
    print("-----------------------------------------------")
    print(preprocessing_hyperparameter_string)

    X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, event_lengths_train = preprocess_per_batch_and_write(X_train_dfs, y_train_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, train_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing, dry_run)
    
    for method_name in methods:
        print("-----------------------------------------------")
        print("Now training: " + method_name)
        print("-----------------------------------------------")
        all_hyperparameters = hyperparameter_dict[method_name]
        hyperparameter_list = list(ParameterGrid(all_hyperparameters))
        
        for hyperparameters in hyperparameter_list:
            hyperparameter_string = str(hyperparameters)
            print(hyperparameter_string)
            
            model = methods[method_name](model_folder, preprocessing_hash, **hyperparameters)
            model_name = model.method_name
            hyperparameter_hash = model.get_hyperparameter_hash()
            hyperparameter_hash_filename = model.get_filename()
            
            scores_path = os.path.join(base_scores_path, preprocessing_hash)
            predictions_path = os.path.join(base_predictions_path, preprocessing_hash)
            intermediates_path = os.path.join(base_intermediates_path, preprocessing_hash)
            fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, model_name, preprocessing_hash)
            PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, model_name, preprocessing_hash)
            minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, model_name, preprocessing_hash)
            
            full_model_path = model.get_full_model_path()
            full_metric_path = os.path.join(fscore_path, hyperparameter_hash+".csv")
            full_table_path = os.path.join(PRFAUC_table_path, hyperparameter_hash+".csv")
            full_minmax_path = os.path.join(minmax_stats_path, hyperparameter_hash+".csv")
            
            if training_overwrite or not os.path.exists(full_model_path) or not os.path.exists(full_metric_path) or not os.path.exists(full_table_path) or not os.path.exists(full_minmax_path):
                
                y_train_scores_dfs, y_train_predictions_dfs = model.fit_transform_predict(X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, base_scores_path=scores_path, base_predictions_path=predictions_path, base_intermediates_path=intermediates_path, overwrite=training_overwrite, fit=True, dry_run=dry_run, verbose=verbose)
    
                metric = cutoff_averaged_f_beta(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
                
                minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_train_dfs_preprocessed, y_train_dfs_preprocessed, y_train_predictions_dfs, load_column="S_original")
                absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
                PRFAUC_table = calculate_PRFAUC_table(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
                
                if not dry_run:

                    save_metric(metric, fscore_path, hyperparameter_hash)
                    save_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_hash)
                    save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_hash)
                    
                
            else:
                print("Model already evaluated, loading results instead:")
                metric = load_metric(fscore_path, hyperparameter_hash)
                PRFAUC_table = load_table(PRFAUC_table_path, hyperparameter_hash)
                minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_hash)
                
                #check if loaded model has saved thresholds for correct optimization cutoff set:
                #-Not implemented specifically for ensembles, as only non-ensembles need to be optimized for all cutoffs at once:
                if not hasattr(model, "is_ensemble"):
                    if not model.check_cutoffs(all_cutoffs):
                        print("Loaded model has wrong cutoffs, recalculating thresholds...")
                        model.used_cutoffs = all_cutoffs
                        model.calculate_and_set_thresholds(all_cutoffs)
                    
            model.report_thresholds()
                        
            if report_metrics_and_stats:
                print_metrics_and_stats(metric, minmax_stats, PRFAUC_table)
                
            if not dry_run:
                #save metric to database for easy querying:
                db_cursor.execute("INSERT OR REPLACE INTO experiment_results VALUES (?, ?, ?, ?, ?, ?, ?)", (preprocessing_hash, hyperparameter_hash, method_name, which_split, jsonpickle.encode(preprocessing_hyperparameters, keys=True), jsonpickle.encode(hyperparameters, keys=True), metric))
                db_connection.commit()
            
            if model_test_run:
                break
#%% dry_run check

if dry_run:
    raise RuntimeError("Dry run is not implemented past training.")
#%% Validation
#%% load Validation data
which_split = "Validation"
print("-----------------------------------------------")
print("Split: Validation")
print("-----------------------------------------------")
X_val_dfs, y_val_dfs, X_val_files = load_batch(data_folder, which_split)

val_file_names = X_val_files

base_scores_path = os.path.join(score_folder, which_split)
base_predictions_path = os.path.join(predictions_folder, which_split)
base_intermediates_path = os.path.join(intermediates_folder, which_split)
    
#%% Preprocess val data and run algorithms:
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_hyperparameter_list = list(ParameterGrid(all_preprocessing_hyperparameters))
for preprocessing_hyperparameters in preprocessing_hyperparameter_list:
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()
    
    print("-----------------------------------------------")
    print("Now preprocessing: ")
    print("-----------------------------------------------")
    print(preprocessing_hyperparameter_string)
    X_val_dfs_preprocessed, y_val_dfs_preprocessed, label_filters_for_all_cutoffs_val, event_lengths_val = preprocess_per_batch_and_write(X_val_dfs, y_val_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, val_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing, dry_run)
    
    for method_name in methods:
        print("-----------------------------------------------")
        print("Now validating: " + method_name)
        print("-----------------------------------------------")
        all_hyperparameters = hyperparameter_dict[method_name]
        hyperparameter_list = list(ParameterGrid(all_hyperparameters))
        
        for hyperparameters in hyperparameter_list:
            hyperparameter_string = str(hyperparameters)
            print(hyperparameter_string)
            
            
            model = methods[method_name](model_folder, preprocessing_hash, **hyperparameters)
            model_name = model.method_name
            hyperparameter_hash = model.get_hyperparameter_hash()
            hyperparameter_hash_filename = model.get_filename()
            
            scores_path = os.path.join(base_scores_path, preprocessing_hash)
            predictions_path = os.path.join(base_predictions_path, preprocessing_hash)
            intermediates_path = os.path.join(base_intermediates_path, preprocessing_hash)
            fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, model_name, preprocessing_hash)
            PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, model_name, preprocessing_hash)
            minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, model_name, preprocessing_hash)
            
            full_metric_path = os.path.join(fscore_path, hyperparameter_hash+".csv")
            full_table_path = os.path.join(PRFAUC_table_path, hyperparameter_hash+".csv")
            full_minmax_path = os.path.join(minmax_stats_path, hyperparameter_hash+".csv")
            
            if validation_overwrite or not os.path.exists(full_metric_path) or not os.path.exists(full_table_path) or not os.path.exists(full_minmax_path):
                
                y_val_scores_dfs, y_val_predictions_dfs = model.transform_predict(X_val_dfs_preprocessed, y_val_dfs_preprocessed, label_filters_for_all_cutoffs_val, base_scores_path=scores_path, base_predictions_path=predictions_path, base_intermediates_path=intermediates_path, overwrite=validation_overwrite, verbose=verbose)
    
                metric = cutoff_averaged_f_beta(y_val_dfs_preprocessed, y_val_predictions_dfs, label_filters_for_all_cutoffs_val, beta)
                
                minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_val_dfs_preprocessed, y_val_dfs_preprocessed, y_val_predictions_dfs, load_column="S_original")
                absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
                PRFAUC_table = calculate_PRFAUC_table(y_val_dfs_preprocessed, y_val_predictions_dfs, label_filters_for_all_cutoffs_val, beta)
                
                if not dry_run:
                    save_metric(metric, fscore_path, hyperparameter_hash)
                    save_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_hash)
                    save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_hash)
                    
            else:
                print("Model already evaluated, loading results instead:")
                metric = load_metric(fscore_path, hyperparameter_hash)
                PRFAUC_table = load_table(PRFAUC_table_path, hyperparameter_hash)
                minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_hash)
                
                        
            if report_metrics_and_stats:
                print_metrics_and_stats(metric, minmax_stats, PRFAUC_table)
                
            if not dry_run:
                #save metric to database for easy querying:
                db_cursor.execute("INSERT OR REPLACE INTO experiment_results VALUES (?, ?, ?, ?, ?, ?, ?)", (preprocessing_hash, hyperparameter_hash, method_name, which_split, jsonpickle.encode(preprocessing_hyperparameters, keys=True), jsonpickle.encode(hyperparameters, keys=True), metric))
                db_connection.commit()
                
            if model_test_run:
                break


#%% Test
# Preprocess Test data and run algorithms:
# Peprocess entire batch
# Save preprocessed data for later recalculations

which_split = "Test"
print("-----------------------------------------------")
print("Split: Test")
print("-----------------------------------------------")
X_test_dfs, y_test_dfs, X_test_files = load_batch(data_folder, which_split)
test_file_names = X_test_files

base_scores_path = os.path.join(score_folder, which_split)
base_predictions_path = os.path.join(predictions_folder, which_split)
base_intermediates_path = os.path.join(intermediates_folder, which_split)

best_hyperparameters = {}
best_preprocessing_hyperparameters = {}
for method_name in methods:
    print("-----------------------------------------------")
    print("Now Testing: " + method_name)
    print("-----------------------------------------------")
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

    
    (preprocessing_hash, hyperparameter_hash, _, _, preprocessing_hyperparameter_string_pickle, hyperparameter_string_pickle, _) = next(best_model_entry)
    
    best_hyperparameters[method_name] = jsonpickle.decode(hyperparameter_string_pickle, keys=True)
    best_preprocessing_hyperparameters[method_name] = jsonpickle.decode(preprocessing_hyperparameter_string_pickle, keys=True)

    preprocessing_hyperparameters = best_preprocessing_hyperparameters[method_name]
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()
    
    print("-----------------------------------------------")
    print("Now preprocessing: ")
    print("-----------------------------------------------")
    print(preprocessing_hyperparameter_string)
    X_test_dfs_preprocessed, y_test_dfs_preprocessed, label_filters_for_all_cutoffs_test, event_lengths_test = preprocess_per_batch_and_write(X_test_dfs, y_test_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, test_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing, dry_run)

 
    hyperparameters = best_hyperparameters[method_name] 
    hyperparameter_string = str(hyperparameters)
    print(hyperparameter_string)
    
    model = methods[method_name](model_folder, preprocessing_hash, **hyperparameters)
    model_name = model.method_name
    hyperparameter_hash = model.get_hyperparameter_hash()
    hyperparameter_hash_filename = model.get_filename()
    
    scores_path = os.path.join(base_scores_path, preprocessing_hash)
    predictions_path = os.path.join(base_predictions_path, preprocessing_hash)
    intermediates_path = os.path.join(base_intermediates_path, preprocessing_hash)
    fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, model_name, preprocessing_hash)
    PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, model_name, preprocessing_hash)
    minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, model_name, preprocessing_hash)
    
    PRF_mean_table_path = os.path.join(metric_folder, "PRF_mean_table", which_split, model_name, preprocessing_hash)
    PRF_std_table_path = os.path.join(metric_folder, "PRF_std_table", which_split, model_name, preprocessing_hash)
    avg_fbeta_mean_path = os.path.join(metric_folder, "bootstrap_mean_F"+str(beta), which_split, model_name, preprocessing_hash)
    avg_fbeta_std_path = os.path.join(metric_folder, "bootstrap_std_F"+str(beta), which_split, model_name, preprocessing_hash)

    full_metric_path = os.path.join(fscore_path, hyperparameter_hash+".csv")
    full_table_path = os.path.join(PRFAUC_table_path, hyperparameter_hash+".csv")
    full_minmax_path = os.path.join(minmax_stats_path, hyperparameter_hash+".csv")
    
    if testing_overwrite or not os.path.exists(full_metric_path) or not os.path.exists(full_table_path) or not os.path.exists(full_minmax_path):
        
        y_test_scores_dfs, y_test_predictions_dfs = model.transform_predict(X_test_dfs_preprocessed, y_test_dfs_preprocessed, label_filters_for_all_cutoffs_test, base_scores_path=scores_path, base_predictions_path=predictions_path, base_intermediates_path=intermediates_path, overwrite=testing_overwrite, verbose=verbose)

        metric = cutoff_averaged_f_beta(y_test_dfs_preprocessed, y_test_predictions_dfs, label_filters_for_all_cutoffs_test, beta)
        
        minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_test_dfs_preprocessed, y_test_dfs_preprocessed, y_test_predictions_dfs, load_column="S_original")
        absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
        PRFAUC_table = calculate_PRFAUC_table(y_test_dfs_preprocessed, y_test_predictions_dfs, label_filters_for_all_cutoffs_test, beta)
        
        if bootstrap_validation:
            PRF_mean_table, PRF_std_table, avg_fbeta_mean, avg_fbeta_std = calculate_bootstrap_stats(y_test_dfs_preprocessed, y_test_predictions_dfs, label_filters_for_all_cutoffs_test, beta, bootstrap_iterations=bootstrap_iterations)
            
        if not dry_run:
            save_metric(metric, fscore_path, hyperparameter_hash)
            save_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_hash)
            save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_hash)
            
            if bootstrap_validation:
                save_table(PRF_mean_table, PRF_mean_table_path, hyperparameter_hash)
                save_table(PRF_std_table, PRF_std_table_path, hyperparameter_hash)
                save_metric(avg_fbeta_mean, avg_fbeta_mean_path, hyperparameter_hash)
                save_metric(avg_fbeta_std, avg_fbeta_std_path, hyperparameter_hash)
                
    else:
        print("Model already evaluated, loading results instead:")
        metric = load_metric(fscore_path, hyperparameter_hash)
        PRFAUC_table = load_table(PRFAUC_table_path, hyperparameter_hash)
        minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_hash)
    
        if bootstrap_validation:
            PRF_mean_table = load_table(PRF_mean_table_path, hyperparameter_hash)
            PRF_std_table = load_table(PRF_std_table_path, hyperparameter_hash)
            avg_fbeta_mean = load_metric(avg_fbeta_mean_path, hyperparameter_hash)
            avg_fbeta_std = load_metric(avg_fbeta_std_path, hyperparameter_hash)
    
    
    if report_metrics_and_stats:
        print_metrics_and_stats(metric, minmax_stats, PRFAUC_table)
        
        if bootstrap_validation:
            print("Bootstrap results:")
            print(bootstrap_stats_to_printable(PRF_mean_table, PRF_std_table))
            print("bootstrapped F"+str(beta))
            print("{0:.4f}".format(avg_fbeta_mean)+"±"+"{0:.4f}".format(avg_fbeta_std))
            
    if not dry_run:
        #save metric to database for easy querying:
        db_cursor.execute("INSERT OR REPLACE INTO experiment_results VALUES (?, ?, ?, ?, ?, ?, ?)", (preprocessing_hash, hyperparameter_hash, method_name, which_split, jsonpickle.encode(preprocessing_hyperparameters), jsonpickle.encode(hyperparameters), metric))
        db_connection.commit()
        

