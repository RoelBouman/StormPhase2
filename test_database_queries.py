#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:28:46 2023

@author: rbouman
"""
#%% Load packages
import os
import sqlite3
import jsonpickle

DBFILE = "experiment_results.db"
database_exists = os.path.exists(DBFILE)

db_connection = sqlite3.connect(DBFILE) # implicitly creates DBFILE if it doesn't exist
db_cursor = db_connection.cursor()
if not database_exists:
    db_cursor.execute("CREATE TABLE experiment_results(preprocessing_hash, hyperparameter_hash, method, which_split, preprocesing_hyperparameters, method_hyperparameters, metric, PRIMARY KEY (preprocessing_hash, hyperparameter_hash, method, which_split))")
    
    
# print("-- Querying the database")
# for row in db_cursor.execute("SELECT preprocessing_hash, hyperparameter_hash, method, which_split, preprocesing_hyperparameters, method_hyperparameters, metric FROM experiment_results"):
#     print(row)
    
method_name = "SingleThresholdSP"
which_split = "Train"

best_hyperparameters = {}
best_preprocessing_hyperparameters = {}
#test = db_cursor.execute("SELECT method, which_split, preprocesing_hyperparameters, method_hyperparameters, metric FROM experiment_results WHERE method={} AND which_split={}".format(method_name, which_split))
best_model_entry = db_cursor.execute("SELECT e.* FROM experiment_results e WHERE e.metric = (SELECT MAX(metric)FROM experiment_results WHERE method = (?) AND which_split = (?))", (method_name, which_split))

(preprocessing_hash, hyperparameter_hash, _, _, preprocessing_hyperparameter_string, hyperparameter_string, _) = next(best_model_entry)

best_hyperparameters[method_name] = jsonpickle.decode(hyperparameter_string)
best_preprocessing_hyperparameters[method_name] = jsonpickle.decode(preprocessing_hyperparameter_string)

