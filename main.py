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

from src.preprocess import preprocess_per_batch_and_write
from src.io_functions import save_dataframe_list, save_metric, save_PRFAUC_table, save_minmax_stats
from src.io_functions import load_batch, load_metric, load_PRFAUC_table, load_minmax_stats
#from src.io_functions import print_count_nan
from src.evaluation import f_beta, cutoff_averaged_f_beta, calculate_unsigned_absolute_and_relative_stats, calculate_PRFAUC_table

from src.reporting_functions import print_metrics_and_stats

from src.plot_functions import plot_predictions

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

training_overwrite = True 
testing_overwrite = False
validation_overwrite = False

#%% set up database

DBFILE = "experiment_results.db"
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()
if not database_exists:
    db_cursor.execute("CREATE TABLE experiment_results(preprocessing_hash, hyperparameter_hash, method, which_split, preprocessing_hyperparameters, method_hyperparameters, metric, PRIMARY KEY (preprocessing_hash, hyperparameter_hash, method, which_split))")
    
#%% define hyperparemeters for preprocessing

all_preprocessing_hyperparameters = {'subsequent_nr': [5], 'lin_fit_quantiles': [(10, 90)]}

#%% define hyperparameters per method:

SingleThresholdSPC_hyperparameters = {"quantiles":[(5,95), (10,90), (15, 85), (20,80), (25,75)]}

DoubleThresholdSPC_hyperparameters = {"quantiles":[(5,95), (10,90), (15, 85), (20,80), (25,75)]}

SingleThresholdIF_hyperparameters = {"n_estimators": [1000], "forest_per_station":[True, False]}

SingleThresholdBS_hyperparameters = {"beta": [0.005, 0.008, 0.12, 0.015], "model": ['l1'], 'min_size': [100], "jump": [10], "quantiles": [(5,95)], "scaling": [True], "penalty": ['fused_lasso']}

DoubleThresholdBS_hyperparameters = {"beta": [0.005, 0.008, 0.12, 0.015], "model": ['l1'], 'min_size': [100], "jump": [10], "quantiles": [(5,95)], "scaling": [True], "penalty": ['fused_lasso']}

NaiveStackEnsemble_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]], "method_hyperparameter_dict_list":[[{'beta':0.12, 'model':'l1','min_size':100, 'jump':10, 'quantiles':(5,95), 'scaling':True, 'penalty':'fused_lasso'},{'quantiles': (5, 95)}]], "all_cutoffs":[all_cutoffs]}

StackEnsemble_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]], "method_hyperparameter_dict_list":[[{'beta':0.12, 'model':'l1','min_size':100, 'jump':10, 'quantiles':(5,95), 'scaling':True, 'penalty':'fused_lasso'},{'quantiles': (5, 95)}]], "cutoffs_per_method":[[all_cutoffs[2:], all_cutoffs[:2]]]}


#%% testrun define methods:

#methods = {"SingleThresholdSPC":SingleThresholdStatisticalProcessControl, "DoubleThresholdSPC": DoubleThresholdStatisticalProcessControl,
#           "SingleThresholdIF":SingleThresholdIsolationForest}
#hyperparameter_dict = {"SingleThresholdSPC":SingleThresholdSPC_hyperparameters, "DoubleThresholdSPC":DoubleThresholdSPC_hyperparameters,
#                       "SingleThresholdIF":SingleThresholdIF_hyperparameters}

#methods = {"SingleThresholdIF":SingleThresholdIsolationForest}
#hyperparameter_dict = {"SingleThresholdIF":SingleThresholdIF_hyperparameters}

# DoubleThresholdSPC_hyperparameters = {"quantiles":[(5,95)], "used_cutoffs":[all_cutoffs]}
# SingleThresholdSPC_hyperparameters = {"quantiles":[(5,95)], "used_cutoffs":[all_cutoffs]}
# methods = {"SingleThresholdSPC":SingleThresholdStatisticalProcessControl, "DoubleThresholdSPC":DoubleThresholdStatisticalProcessControl}
# hyperparameter_dict = {"SingleThresholdSPC":SingleThresholdSPC_hyperparameters, "DoubleThresholdSPC":DoubleThresholdSPC_hyperparameters}

# SingleThresholdBS_hyperparameters = {"beta": [0.12], "model": ['l1'], 'min_size': [100], "jump": [10], "quantiles": [(5,95)], "scaling": [True], "penalty": ['fused_lasso']}
# DoubleThresholdBS_hyperparameters = {"beta": [0.12], "model": ['l1'], 'min_size': [100], "jump": [10], "quantiles": [(5,95)], "scaling": [True], "penalty": ['fused_lasso']}
# methods = {"SingleThresholdBS":SingleThresholdBinarySegmentation, "DoubleThresholdBS":DoubleThresholdBinarySegmentation}
# hyperparameter_dict = {"SingleThresholdBS":SingleThresholdBS_hyperparameters, "DoubleThresholdBS":DoubleThresholdBS_hyperparameters}

# NaiveStackEnsemble_hyperparameters = {"method_classes":[[DoubleThresholdStatisticalProcessControl, SingleThresholdStatisticalProcessControl]], "method_hyperparameter_dict_list":[[{'quantiles': (5, 94)},{'quantiles': (5, 94)}]], "all_cutoffs":[all_cutoffs]}
# methods = {"NaiveStackEnsemble":NaiveStackEnsemble}
# hyperparameter_dict = {"NaiveStackEnsemble":NaiveStackEnsemble_hyperparameters}


#For IF, pass sequence of dicts to avoid useless hyperparam combos (such as scaling=True, forest_per_station=True)
SingleThresholdIF_hyperparameters = [{"n_estimators": [1000], "forest_per_station":[True], "scaling":[False]}, {"n_estimators": [1000], "forest_per_station":[False], "scaling":[True], "quantiles":[(10,90)]}]
                                     
SingleThresholdSPC_hyperparameters = {"quantiles":[(10,90)], "used_cutoffs":[all_cutoffs]}
SingleThresholdBS_hyperparameters = {"beta": [0.12], "model": ['l1'], 'min_size': [100], "jump": [10], "quantiles": [(25,75)], "scaling": [True], "penalty": ['fused_lasso'], "reference_point":["longest_mean"]}
SingleThresholdBS_SingleThresholdSPC_hyperparameters = {"method_classes":[[SingleThresholdBinarySegmentation, SingleThresholdStatisticalProcessControl]], "method_hyperparameter_dict_list":[[{'beta':0.12, 'model':'l1','min_size':100, 'jump':10, 'quantiles':(5,95), 'scaling':True, 'penalty':'fused_lasso'},{'quantiles': (5, 95)}]], "cutoffs_per_method":[[all_cutoffs[2:], all_cutoffs[:2]]]}

#methods = {"SingleThresholdBS":SingleThresholdBinarySegmentation}
#methods = {"SingleThresholdBS":SingleThresholdBinarySegmentation, "SingleThresholdSPC":SingleThresholdStatisticalProcessControl, "SingleThresholdBS+SingleThresholdSPC":StackEnsemble, "SingleThresholdIF":SingleThresholdIsolationForest}
methods = {"SingleThresholdSPC":SingleThresholdStatisticalProcessControl}
hyperparameter_dict = {"SingleThresholdBS":SingleThresholdBS_hyperparameters, "SingleThresholdSPC":SingleThresholdSPC_hyperparameters, "SingleThresholdBS+SingleThresholdSPC":SingleThresholdBS_SingleThresholdSPC_hyperparameters, "SingleThresholdIF":SingleThresholdIF_hyperparameters}
#%% Preprocess Train data and run algorithms:
# Peprocess entire batch
# Save preprocessed data for later recalculations

# load Train data
which_split = "Train"

print("Split: Train")
X_train_dfs, y_train_dfs, X_train_files = load_batch(data_folder, which_split)

train_file_names = X_train_files

preprocessing_hyperparameter_list = list(ParameterGrid(all_preprocessing_hyperparameters))
for preprocessing_hyperparameters in preprocessing_hyperparameter_list:
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()
    
    print("Now preprocessing: ")
    print(preprocessing_hyperparameter_string)
    X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, event_lengths_train = preprocess_per_batch_and_write(X_train_dfs, y_train_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, train_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing)
    
    for method_name in methods:
        print("Now training: " + method_name)
        all_hyperparameters = hyperparameter_dict[method_name]
        hyperparameter_list = list(ParameterGrid(all_hyperparameters))
        
        for hyperparameters in hyperparameter_list:
            hyperparameter_string = str(hyperparameters)
            print(hyperparameter_string)
            
            ### NEW
            model = methods[method_name](model_folder, preprocessing_hash, **hyperparameters, score_function=score_function)
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
                
                y_train_scores_dfs, y_train_predictions_dfs = model.fit_transform_predict(X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, base_scores_path=base_scores_path, base_predictions_path=base_predictions_path, overwrite=training_overwrite)
    
                metric = cutoff_averaged_f_beta(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
                
                minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_train_dfs_preprocessed, y_train_dfs_preprocessed, y_train_predictions_dfs, load_column="S_original")
                absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
                PRFAUC_table = calculate_PRFAUC_table(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)
                
                save_dataframe_list(y_train_scores_dfs, train_file_names, os.path.join(scores_path, "stations"), overwrite=training_overwrite)
                save_dataframe_list(y_train_predictions_dfs, train_file_names, os.path.join(predictions_path, "stations"), overwrite=training_overwrite)
                
                save_metric(metric, fscore_path, hyperparameter_hash)
                save_PRFAUC_table(PRFAUC_table, PRFAUC_table_path, hyperparameter_hash)
                save_minmax_stats(minmax_stats, minmax_stats_path, hyperparameter_hash)
                
                #save metric to database for easy querying:
                db_cursor.execute("INSERT OR REPLACE INTO experiment_results VALUES (?, ?, ?, ?, ?, ?, ?)", (preprocessing_hash, hyperparameter_hash, method_name, which_split, jsonpickle.encode(preprocessing_hyperparameters), jsonpickle.encode(hyperparameters), metric))
                db_connection.commit()
                
                #save_model(model, model_path, hyperparameter_string)
                model.save_model()
                
            else:
                print("Model already evaluated, loading results instead:")
                metric = load_metric(fscore_path, hyperparameter_hash)
                PRFAUC_table = load_PRFAUC_table(PRFAUC_table_path, hyperparameter_hash)
                minmax_stats = load_minmax_stats(minmax_stats_path, hyperparameter_hash)
                
                #Model is instead loaded at model inititiation 
                #model.load_model(model_path, hyperparameter_string)
            
                #check if loaded model has saved thresholds for correct optimization cutoff set:
                #-Not implemented specifically for ensembles, as only non-ensembles need to be optimized for all cutoffs at once:
                if not hasattr(model, "is_ensemble"):
                    if not model.check_cutoffs(all_cutoffs):
                        print("Loaded model has wrong cutoffs, recalculating thresholds...")
                        model.used_cutoffs = all_cutoffs
                        model.calculate_thresholds(all_cutoffs, score_function)
                    
            model.report_thresholds()
                        
            if report_metrics_and_stats:
                print_metrics_and_stats(metric, minmax_stats, PRFAUC_table)

#%% Test
#%% load Test data
which_split = "Test"
print("Split: Test")
X_test_dfs, y_test_dfs, X_test_files = load_batch(data_folder, which_split)

test_file_names = X_test_files
    
#%% Preprocess Test data and run algorithms:
# Peprocess entire batch
# Save preprocessed data for later recalculations

preprocessing_hyperparameter_list = list(ParameterGrid(all_preprocessing_hyperparameters))
for preprocessing_hyperparameters in preprocessing_hyperparameter_list:
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()
    
    print("Now preprocessing: ")
    print(preprocessing_hyperparameter_string)
    X_test_dfs_preprocessed, y_test_dfs_preprocessed, label_filters_for_all_cutoffs_test, event_lengths_test = preprocess_per_batch_and_write(X_test_dfs, y_test_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, test_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing)
    
    for method_name in methods:
        print("Now testing: " + method_name)
        all_hyperparameters = hyperparameter_dict[method_name]
        hyperparameter_list = list(ParameterGrid(all_hyperparameters))
        
        for hyperparameters in hyperparameter_list:
            hyperparameter_string = str(hyperparameters)
            print(hyperparameter_string)
            
            ### NEW
            model = methods[method_name](model_folder, preprocessing_hash, **hyperparameters, score_function=score_function)
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


#%% Validation
#%% load Validation data


#%% Preprocess Val data and run algorithms:
# Peprocess entire batch
# Save preprocessed data for later recalculations

which_split = "Validation"
print("Split: Validation")
X_val_dfs, y_val_dfs, X_val_files = load_batch(data_folder, which_split)
val_file_names = X_val_files

best_hyperparameters = {}
best_preprocessing_hyperparameters = {}
for method_name in methods:
    print("Now validating: " + method_name)
    #find best preprocessing and method hyperparameters:

    #Some SQL query:
    
    best_model_entry = db_cursor.execute("SELECT e.* FROM experiment_results e WHERE e.metric = (SELECT MAX(metric)FROM experiment_results WHERE method = (?) AND which_split = (?))", (method_name, "Test"))
    
    (preprocessing_hash, hyperparameter_hash, _, _, preprocessing_hyperparameter_string_pickle, hyperparameter_string_pickle, _) = next(best_model_entry)
    
    best_hyperparameters[method_name] = jsonpickle.decode(hyperparameter_string_pickle)
    best_preprocessing_hyperparameters[method_name] = jsonpickle.decode(preprocessing_hyperparameter_string_pickle)

    preprocessing_hyperparameters = best_preprocessing_hyperparameters[method_name]
    preprocessing_hyperparameter_string = str(preprocessing_hyperparameters)
    preprocessing_hash = sha256(preprocessing_hyperparameter_string.encode("utf-8")).hexdigest()
    
    print("Now preprocessing: ")
    print(preprocessing_hyperparameter_string)
    X_val_dfs_preprocessed, y_val_dfs_preprocessed, label_filters_for_all_cutoffs_val, event_lengths_val = preprocess_per_batch_and_write(X_val_dfs, y_val_dfs, intermediates_folder, which_split, preprocessing_overwrite, write_csv_intermediates, val_file_names, all_cutoffs, preprocessing_hyperparameters, preprocessing_hash, remove_missing)

 
    hyperparameters = best_hyperparameters[method_name] 
    hyperparameter_string = str(hyperparameters)
    print(hyperparameter_string)
    
    ### NEW
    model = methods[method_name](model_folder, preprocessing_hash, **hyperparameters, score_function=score_function)
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

