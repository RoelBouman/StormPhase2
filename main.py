#%% Load packages
import os
import sqlite3
import jsonpickle

import numpy as np
from sklearn.model_selection import ParameterGrid
#import pandas as pd
from hashlib import sha256

from src.methods import SingleThresholdStatisticalProcessControl
from src.methods import IndependentDoubleThresholdStatisticalProcessControl


from src.methods import SingleThresholdIsolationForest

from src.methods import SingleThresholdBinarySegmentation
from src.methods import IndependentDoubleThresholdBinarySegmentation

from src.methods import StackEnsemble
from src.methods import NaiveStackEnsemble

from src.preprocess import preprocess_per_batch_and_write
from src.io_functions import save_dataframe_list, save_metric, save_PRFAUC_table, save_minmax_stats
from src.io_functions import load_batch, load_metric, load_PRFAUC_table, load_minmax_stats
#from src.io_functions import print_count_nan
from src.evaluation import cutoff_averaged_f_beta, calculate_unsigned_absolute_and_relative_stats, calculate_PRFAUC_table

from src.reporting_functions import print_metrics_and_stats

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
uncertain_filter = [4,5] #which labels not to include in evaluation

beta = 1.5
report_metrics_and_stats = True

remove_missing = True

write_csv_intermediates = True

preprocessing_overwrite = False #if set to True, overwrite previous preprocessed data

training_overwrite = False 
validation_overwrite = False
testing_overwrite = False

dry_run = True

#%% set up database

DBFILE = dataset+"_experiment_results.db"
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()
if not database_exists:
    db_cursor.execute("CREATE TABLE experiment_results(preprocessing_hash, hyperparameter_hash, method, which_split, preprocessing_hyperparameters, method_hyperparameters, metric, PRIMARY KEY (preprocessing_hash, hyperparameter_hash, method, which_split))")
    
#%% define hyperparemeters for preprocessing

all_preprocessing_hyperparameters = {'subsequent_nr': [5], 'lin_fit_quantiles': [(10, 90)]}

#%% define hyperparameters per method:



#For IF, pass sequence of dicts to avoid useless hyperparam combos (such as scaling=True, forest_per_station=True)
SingleThresholdIF_hyperparameters = [{"n_estimators": [1000], "forest_per_station":[True], "scaling":[False], "score_function_kwargs":[{"beta":beta}]}, {"n_estimators": [1000], "forest_per_station":[False], "scaling":[True], "quantiles":[(5,95), (10,90), (15, 85), (20,80), (25,75)], "score_function_kwargs":[{"beta":beta}]}]
SingleThresholdSPC_hyperparameters = {"quantiles":[(5,95), (10,90), (15, 85), (20,80), (25,75)], "score_function_kwargs":[{"beta":beta}]}
SingleThresholdBS_hyperparameters = {"beta": [0.005, 0.008, 0.015, 0.05, 0.08, 0.12], "model": ['l1'], 'min_size': [50, 100, 200], "jump": [10], "quantiles": [(5,95), (10,90), (15, 85), (20,80)], "scaling": [True], "penalty": ['fused_lasso'], "reference_point":["mean", "median", "longest_median", "longest_mean"], "score_function_kwargs":[{"beta":beta}]}


SingleThresholdBS_SingleThresholdSPC_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]], "method_hyperparameter_dict_list":[[{'beta':0.12, 'model':'l1','min_size':100, 'jump':10, 'quantiles':(5,95), 'scaling':True, 'penalty':'fused_lasso'},{'quantiles': (5, 95)}]], "cutoffs_per_method":[[all_cutoffs[2:], all_cutoffs[:2]]]}
Naive_SingleThresholdBS_SingleThresholdSPC_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]], "method_hyperparameter_dict_list":[[{'beta':0.12, 'model':'l1','min_size':100, 'jump':10, 'quantiles':(5,95), 'scaling':True, 'penalty':'fused_lasso'},{'quantiles': (5, 95)}]], "all_cutoffs":[all_cutoffs]}


DoubleThresholdSPC_hyperparameters = SingleThresholdSPC_hyperparameters
DoubleThresholdBS_hyperparameters = SingleThresholdBS_hyperparameters
DoubleThresholdBS_DoubleThresholdSPC_hyperparameters = SingleThresholdBS_SingleThresholdSPC_hyperparameters
Naive_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters = Naive_SingleThresholdBS_SingleThresholdSPC_hyperparameters


IndependentDoubleThresholdSPC_hyperparameters = SingleThresholdSPC_hyperparameters
IndependentDoubleThresholdBS_hyperparameters = SingleThresholdBS_hyperparameters
#%% define methods:
SingleThresholdBS_hyperparameters = {"beta": [0.015], "model": ['l1'], 'min_size': [50], "jump": [10], "quantiles": [(15, 85)], "scaling": [True], "penalty": ['fused_lasso'], "reference_point":["mean", "median", "longest_median", "longest_mean"], "score_function_kwargs":[{"beta":beta}]}
DoubleThresholdBS_hyperparameters = SingleThresholdBS_hyperparameters
IndependentDoubleThresholdBS_hyperparameters = SingleThresholdBS_hyperparameters
    

SingleThresholdSPC_hyperparameters = {"quantiles":[(10,90)], "score_function_kwargs":[{"beta":beta}]}
DoubleThresholdSPC_hyperparameters = SingleThresholdSPC_hyperparameters
IndependentDoubleThresholdSPC_hyperparameters = SingleThresholdSPC_hyperparameters

# Naive_SingleThresholdBS_IndependentDoubleThresholdSPC_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, IndependentDoubleThresholdStatisticalProcessControl]], "method_hyperparameter_dict_list":[[list(ParameterGrid(SingleThresholdBS_hyperparameters))[2],list(ParameterGrid(SingleThresholdSPC_hyperparameters))[0]]], "all_cutoffs":[all_cutoffs]}
# SingleThresholdBS_IndependentDoubleThresholdSPC_hyperparameters = Naive_SingleThresholdBS_IndependentDoubleThresholdSPC_hyperparameters.copy()
# del SingleThresholdBS_IndependentDoubleThresholdSPC_hyperparameters["all_cutoffs"] 
# SingleThresholdBS_IndependentDoubleThresholdSPC_hyperparameters["cutoffs_per_method"] = [[all_cutoffs[2:], all_cutoffs[:2]]]

methods = {"SingleThresholdIF":SingleThresholdIsolationForest,
             #"SingleThresholdBS":SingleThresholdBinarySegmentation, 
            # "SingleThresholdSPC":SingleThresholdStatisticalProcessControl,
           #  "SingleThresholdBS+SingleThresholdSPC":StackEnsemble, 
           #  "Naive-SingleThresholdBS+SingleThresholdSPC":NaiveStackEnsemble, 
             #"DoubleThresholdBS":DoubleThresholdBinarySegmentation, 
             #"DoubleThresholdSPC":DoubleThresholdStatisticalProcessControl, 
            # "DoubleThresholdBS+DoubleThresholdSPC":StackEnsemble, 
            # "Naive-DoubleThresholdBS+DoubleThresholdSPC":NaiveStackEnsemble,
             #"IndependentDoubleThresholdSPC":IndependentDoubleThresholdStatisticalProcessControl,
             #"IndependentDoubleThresholdBS":IndependentDoubleThresholdBinarySegmentation,
             #"Naive-SingleThresholdBS+IndependentDoubleThresholdSPC":NaiveStackEnsemble,
             #"SingleThresholdBS+IndependentDoubleThresholdSPC":StackEnsemble

           }
hyperparameter_dict = {"SingleThresholdIF":SingleThresholdIF_hyperparameters,
                       "SingleThresholdSPC":SingleThresholdSPC_hyperparameters, 
                       "SingleThresholdBS":SingleThresholdBS_hyperparameters, 
                       "SingleThresholdBS+SingleThresholdSPC":SingleThresholdBS_SingleThresholdSPC_hyperparameters, 
                       "Naive-SingleThresholdBS+SingleThresholdSPC":Naive_SingleThresholdBS_SingleThresholdSPC_hyperparameters, 
                       "DoubleThresholdBS":DoubleThresholdBS_hyperparameters, 
                       "DoubleThresholdSPC":DoubleThresholdSPC_hyperparameters, 
                       "DoubleThresholdBS+DoubleThresholdSPC":DoubleThresholdBS_DoubleThresholdSPC_hyperparameters, 
                       "Naive-DoubleThresholdBS+DoubleThresholdSPC":Naive_DoubleThresholdBS_DoubleThresholdSPC_hyperparameters,
                       "IndependentDoubleThresholdSPC":IndependentDoubleThresholdSPC_hyperparameters,
                       "IndependentDoubleThresholdBS":IndependentDoubleThresholdBS_hyperparameters,
                       "Naive-SingleThresholdBS+IndependentDoubleThresholdSPC":Naive_SingleThresholdBS_IndependentDoubleThresholdSPC_hyperparameters,
                       "SingleThresholdBS+IndependentDoubleThresholdSPC":SingleThresholdBS_IndependentDoubleThresholdSPC_hyperparameters
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

preprocessing_hyperparameter_list = list(ParameterGrid(all_preprocessing_hyperparameters))
for preprocessing_hyperparameters in preprocessing_hyperparameter_list:
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()
    
    print("-----------------------------------------------")
    print("Now preprocessing: ")
    print("-----------------------------------------------")
    print(preprocessing_hyperparameter_string)
    X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, event_lengths_train = preprocess_per_batch_and_write(X_train_dfs, y_train_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, train_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing, uncertain_filter, dry_run)
    
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
            
            base_scores_path = os.path.join(score_folder, which_split)
            base_predictions_path = os.path.join(predictions_folder, which_split)
            scores_path = os.path.join(base_scores_path, model_name, preprocessing_hash, hyperparameter_hash)
            predictions_path = os.path.join(base_predictions_path, model_name, preprocessing_hash, hyperparameter_hash)
            fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, model_name, preprocessing_hash)
            PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, model_name, preprocessing_hash)
            minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, model_name, preprocessing_hash)
            
            full_model_path = os.path.join(model_folder, model_name, preprocessing_hash, hyperparameter_hash_filename)
            
            if training_overwrite or not os.path.exists(full_model_path):
                
                y_train_scores_dfs, y_train_predictions_dfs = model.fit_transform_predict(X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, base_scores_path=base_scores_path, base_predictions_path=base_predictions_path, overwrite=training_overwrite, fit=True, dry_run=dry_run)
    
                metric = cutoff_averaged_f_beta(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
                
                minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_train_dfs_preprocessed, y_train_dfs_preprocessed, y_train_predictions_dfs, load_column="S_original")
                absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
                PRFAUC_table = calculate_PRFAUC_table(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
                
                if not dry_run:
                    save_dataframe_list(y_train_scores_dfs, train_file_names, os.path.join(scores_path, "stations"), overwrite=training_overwrite)
                    save_dataframe_list(y_train_predictions_dfs, train_file_names, os.path.join(predictions_path, "stations"), overwrite=training_overwrite)
                    
                    save_metric(metric, fscore_path, hyperparameter_hash)
                    save_PRFAUC_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_hash)
                    save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_hash)
                    
                    #save metric to database for easy querying:
                    db_cursor.execute("INSERT OR REPLACE INTO experiment_results VALUES (?, ?, ?, ?, ?, ?, ?)", (preprocessing_hash, hyperparameter_hash, method_name, which_split, jsonpickle.encode(preprocessing_hyperparameters), jsonpickle.encode(hyperparameters), metric))
                    db_connection.commit()
                
                    model.save_model()
                
            else:
                print("Model already evaluated, loading results instead:")
                metric = load_metric(fscore_path, hyperparameter_hash)
                PRFAUC_table = load_PRFAUC_table(PRFAUC_table_path, hyperparameter_hash)
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
    X_val_dfs_preprocessed, y_val_dfs_preprocessed, label_filters_for_all_cutoffs_val, event_lengths_val = preprocess_per_batch_and_write(X_val_dfs, y_val_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, val_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing, uncertain_filter, dry_run)
    
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
            
            base_scores_path = os.path.join(score_folder, which_split)
            base_predictions_path = os.path.join(predictions_folder, which_split)
            scores_path = os.path.join(base_scores_path, model_name, preprocessing_hash, hyperparameter_hash)
            predictions_path = os.path.join(base_predictions_path, model_name, preprocessing_hash, hyperparameter_hash)
            fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, model_name, preprocessing_hash)
            PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, model_name, preprocessing_hash)
            minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, model_name, preprocessing_hash)
            
            full_metric_path = os.path.join(fscore_path, hyperparameter_hash+".csv")
            
            if validation_overwrite or not os.path.exists(full_metric_path):
                
                y_val_scores_dfs, y_val_predictions_dfs = model.transform_predict(X_val_dfs_preprocessed, y_val_dfs_preprocessed, label_filters_for_all_cutoffs_val, base_scores_path=base_scores_path, base_predictions_path=base_predictions_path, overwrite=validation_overwrite)
    
                metric = cutoff_averaged_f_beta(y_val_dfs_preprocessed, y_val_predictions_dfs, label_filters_for_all_cutoffs_val, beta)
                
                minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_val_dfs_preprocessed, y_val_dfs_preprocessed, y_val_predictions_dfs, load_column="S_original")
                absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
                PRFAUC_table = calculate_PRFAUC_table(y_val_dfs_preprocessed, y_val_predictions_dfs, label_filters_for_all_cutoffs_val, beta)
                
                if not dry_run:
                    save_dataframe_list(y_val_scores_dfs, val_file_names, os.path.join(scores_path, "stations"), overwrite=validation_overwrite)
                    save_dataframe_list(y_val_predictions_dfs, val_file_names, os.path.join(predictions_path, "stations"), overwrite=validation_overwrite)
                    
                    save_metric(metric, fscore_path, hyperparameter_hash)
                    save_PRFAUC_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_hash)
                    save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_hash)
                    
                    #save metric to database for easy querying:
                    db_cursor.execute("INSERT OR REPLACE INTO experiment_results VALUES (?, ?, ?, ?, ?, ?, ?)", (preprocessing_hash, hyperparameter_hash, method_name, which_split, jsonpickle.encode(preprocessing_hyperparameters), jsonpickle.encode(hyperparameters), metric))
                    db_connection.commit()
            else:
                print("Model already evaluated, loading results instead:")
                metric = load_metric(fscore_path, hyperparameter_hash)
                PRFAUC_table = load_PRFAUC_table(PRFAUC_table_path, hyperparameter_hash)
                minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_hash)
                
                #Model is instead loaded at model inititiation 
                #model.load_model(model_path, hyperparameter_string)
                        
            if report_metrics_and_stats:
                print_metrics_and_stats(metric, minmax_stats, PRFAUC_table)


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

best_hyperparameters = {}
best_preprocessing_hyperparameters = {}
for method_name in methods:
    print("-----------------------------------------------")
    print("Now Testing: " + method_name)
    print("-----------------------------------------------")
    #find best preprocessing and method hyperparameters:

    #Some SQL query:
    
    best_model_entry = db_cursor.execute("SELECT e.* FROM experiment_results e WHERE e.metric = (SELECT MAX(metric)FROM experiment_results WHERE method = (?) AND which_split = (?))", (method_name, "Validation"))
    
    (preprocessing_hash, hyperparameter_hash, _, _, preprocessing_hyperparameter_string_pickle, hyperparameter_string_pickle, _) = next(best_model_entry)
    
    best_hyperparameters[method_name] = jsonpickle.decode(hyperparameter_string_pickle)
    best_preprocessing_hyperparameters[method_name] = jsonpickle.decode(preprocessing_hyperparameter_string_pickle)

    preprocessing_hyperparameters = best_preprocessing_hyperparameters[method_name]
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()
    
    print("-----------------------------------------------")
    print("Now preprocessing: ")
    print("-----------------------------------------------")
    print(preprocessing_hyperparameter_string)
    X_test_dfs_preprocessed, y_test_dfs_preprocessed, label_filters_for_all_cutoffs_test, event_lengths_test = preprocess_per_batch_and_write(X_test_dfs, y_test_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, test_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing, uncertain_filter, dry_run)

 
    hyperparameters = best_hyperparameters[method_name] 
    hyperparameter_string = str(hyperparameters)
    print(hyperparameter_string)
    
    model = methods[method_name](model_folder, preprocessing_hash, **hyperparameters)
    model_name = model.method_name
    hyperparameter_hash = model.get_hyperparameter_hash()
    hyperparameter_hash_filename = model.get_filename()
    
    base_scores_path = os.path.join(score_folder, which_split)
    base_predictions_path = os.path.join(predictions_folder, which_split)
    scores_path = os.path.join(base_scores_path, model_name, preprocessing_hash, hyperparameter_hash)
    predictions_path = os.path.join(base_predictions_path, model_name, preprocessing_hash, hyperparameter_hash)
    fscore_path = os.path.join(metric_folder, "F"+str(beta), which_split, model_name, preprocessing_hash)
    PRFAUC_table_path = os.path.join(metric_folder, "PRFAUC_table", which_split, model_name, preprocessing_hash)
    minmax_stats_path = os.path.join(metric_folder, "minmax_stats", which_split, model_name, preprocessing_hash)
    
    full_metric_path = os.path.join(fscore_path, hyperparameter_hash+".csv")
    
    if testing_overwrite or not os.path.exists(full_metric_path):
        
        y_test_scores_dfs, y_test_predictions_dfs = model.transform_predict(X_test_dfs_preprocessed, y_test_dfs_preprocessed, label_filters_for_all_cutoffs_test, base_scores_path=base_scores_path, base_predictions_path=base_predictions_path, overwrite=testing_overwrite)

        metric = cutoff_averaged_f_beta(y_test_dfs_preprocessed, y_test_predictions_dfs, label_filters_for_all_cutoffs_test, beta)
        
        minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_test_dfs_preprocessed, y_test_dfs_preprocessed, y_test_predictions_dfs, load_column="S_original")
        absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
        PRFAUC_table = calculate_PRFAUC_table(y_test_dfs_preprocessed, y_test_predictions_dfs, label_filters_for_all_cutoffs_test, beta)
        
        if bootstrap_validation:
            metric_mean, metric_std, PRFAUC_table_mean, PRFAUC_table_std = calculate_bootstrap_stats
        
        if not dry_run:
            save_dataframe_list(y_test_scores_dfs, test_file_names, os.path.join(scores_path, "stations"), overwrite=testing_overwrite)
            save_dataframe_list(y_test_predictions_dfs, test_file_names, os.path.join(predictions_path, "stations"), overwrite=testing_overwrite)
            
            save_metric(metric, fscore_path, hyperparameter_hash)
            save_PRFAUC_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_hash)
            save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_hash)
            
            #save metric to database for easy querying:
            db_cursor.execute("INSERT OR REPLACE INTO experiment_results VALUES (?, ?, ?, ?, ?, ?, ?)", (preprocessing_hash, hyperparameter_hash, method_name, which_split, jsonpickle.encode(preprocessing_hyperparameters), jsonpickle.encode(hyperparameters), metric))
            db_connection.commit()
    else:
        print("Model already evaluated, loading results instead:")
        metric = load_metric(fscore_path, hyperparameter_hash)
        PRFAUC_table = load_PRFAUC_table(PRFAUC_table_path, hyperparameter_hash)
        minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_hash)
        
        #Model is instead loaded at model inititiation 
        #model.load_model(model_path, hyperparameter_string)
                
    if report_metrics_and_stats:
        print_metrics_and_stats(metric, minmax_stats, PRFAUC_table)

