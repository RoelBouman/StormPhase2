#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:02:32 2023

@author: rbouman
"""
import numpy as np
import pandas as pd

from sklearn.metrics import fbeta_score, precision_score, recall_score

from .helper_functions import filter_label_and_predictions_to_array



def f_beta(precision, recall, beta=1, eps=np.finfo(float).eps):
    
    return(1+beta**2) * (precision*recall)/((beta**2*precision)+recall+eps)

def cutoff_averaged_f_beta(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, beta):
    all_cutoffs = label_filters_for_all_cutoffs[0].keys()

    f_betas = 0
    for cutoffs in all_cutoffs:
        filtered_y, filtered_y_preds = filter_label_and_predictions_to_array(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, cutoffs)
        f_betas += fbeta_score(filtered_y, filtered_y_preds, beta=beta)
        
    return f_betas/len(all_cutoffs)

def calculate_PRF_table(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, beta):
    all_cutoffs = label_filters_for_all_cutoffs[0].keys()

    precisions = []
    recalls = []
    fbetas = []
    
    for cutoffs in all_cutoffs:
        filtered_y, filtered_y_preds = filter_label_and_predictions_to_array(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, cutoffs)
        precisions.append(precision_score(filtered_y, filtered_y_preds))
        recalls.append(recall_score(filtered_y, filtered_y_preds))
        fbetas.append(fbeta_score(filtered_y, filtered_y_preds, beta=beta))
        
    PRF_table = pd.DataFrame(index=all_cutoffs)
    
    PRF_table["precision"] = precisions
    PRF_table["recall"] = recalls
    PRF_table["F"+str(beta)] = fbetas
    
    return PRF_table

def calculate_minmax_stats(X_dfs, y_dfs, y_pred_dfs, load_column="S_original"):
    
    X_mins, X_pred_mins = [], []
    X_maxs, X_pred_maxs = [], []
    
    for X_df, y_df, y_pred_df in zip(X_dfs, y_dfs, y_pred_dfs):
        X_normal = X_df[y_df["label"]==0][load_column]
        X_pred_normal = X_df[y_pred_df["label"]==0][load_column]

        X_mins.append(X_normal.min())
        X_maxs.append(X_normal.max())

        X_pred_mins.append(X_pred_normal.min())
        X_pred_maxs.append(X_pred_normal.max())
        
    return X_mins, X_maxs, X_pred_mins, X_pred_maxs

def calculate_unsigned_absolute_and_relative_stats(X_dfs, y_dfs, y_pred_dfs, load_column="S_original"):
    X_mins, X_maxs, X_pred_mins, X_pred_maxs = calculate_minmax_stats(X_dfs, y_dfs, y_pred_dfs, load_column)
    
    absolute_min_differences = [np.abs(X_min - X_pred_min) for X_min, X_pred_min in zip(X_mins, X_pred_mins)]
    absolute_max_differences = [np.abs(X_max - X_pred_max) for X_max, X_pred_max in zip(X_maxs, X_pred_maxs)]
    
    relative_min_differences = [(absolute_min_difference)/(X_max-X_min) for X_min, X_max, absolute_min_difference in zip(X_mins, X_maxs, absolute_min_differences)]
    relative_max_differences = [(absolute_max_difference)/(X_max-X_min) for X_min, X_max, absolute_max_difference in zip(X_mins, X_maxs, absolute_max_differences)]

    return absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences

