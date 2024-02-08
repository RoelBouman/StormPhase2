#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:02:32 2023

@author: rbouman
"""
import numpy as np
import pandas as pd
from numba import njit

from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score

from .helper_functions import filter_label_and_predictions_to_array


def f_beta(precision, recall, beta=1, eps=np.finfo(float).eps):
    
    return(1+beta**2) * (precision*recall)/((beta**2*precision)+recall+eps)

def f_beta_from_confmat(fp, tp, fn, beta=1, eps=np.finfo(float).eps):
    
    return (1+beta**2) * (tp) / ((1+beta**2)*tp + (beta**2)*fn + fp + eps)

def cutoff_averaged_f_beta(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, beta):
    all_cutoffs = label_filters_for_all_cutoffs[0].keys()

    f_betas = 0
    for cutoffs in all_cutoffs:
        filtered_y, filtered_y_preds = filter_label_and_predictions_to_array(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, cutoffs)
        f_betas += fbeta_score(filtered_y, filtered_y_preds, beta=beta)
        
    return f_betas/len(all_cutoffs)

def calculate_PRFAUC_table(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, beta):
    all_cutoffs = label_filters_for_all_cutoffs[0].keys()

    precisions = []
    recalls = []
    fbetas = []
    AUCs =[]
    
    for cutoffs in all_cutoffs:
        filtered_y, filtered_y_preds = filter_label_and_predictions_to_array(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, cutoffs)
        precisions.append(precision_score(filtered_y, filtered_y_preds))
        recalls.append(recall_score(filtered_y, filtered_y_preds))
        fbetas.append(fbeta_score(filtered_y, filtered_y_preds, beta=beta))
        AUCs.append(roc_auc_score(filtered_y, filtered_y_preds))
        
    PRFAUC_table = pd.DataFrame(index=all_cutoffs)
    PRFAUC_table.index.name = "Cutoffs"
    
    PRFAUC_table["precision"] = precisions
    PRFAUC_table["recall"] = recalls
    PRFAUC_table["F"+str(beta)] = fbetas
    PRFAUC_table["ROC/AUC"] = AUCs
    
    return PRFAUC_table

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

def calculate_bootstrap_stats(y_dfs, y_pred_dfs, label_filters_for_all_cutoffs, beta, bootstrap_iterations=10000):
    y_arrays = df_list_to_array_list(y_dfs, column="label")
    y_pred_arrays = df_list_to_array_list(y_pred_dfs, column="label")
    label_array_filters_for_all_cutoffs = label_filters_to_array(label_filters_for_all_cutoffs)
    
    metric_mean, metric_std, PRFAUC_table_mean, PRFAUC_table_std = _calculate_bootstrap_stats(y_arrays, y_pred_arrays, label_array_filters_for_all_cutoffs, beta, bootstrap_iterations=bootstrap_iterations)

    return metric_mean, metric_std, PRFAUC_table_mean, PRFAUC_table_std



def calculate_PRFAUC_table_from_arrays(y_arrays, y_pred_arrays, label_array_filters_for_all_cutoffs, beta):
    n_cutoffs = label_array_filters_for_all_cutoffs[0].shape[1]

    precisions = np.zeros((n_cutoffs,))
    recalls = np.zeros((n_cutoffs,))
    fbetas = np.zeros((n_cutoffs,))
    AUCs = np.zeros((n_cutoffs,))
    
    for i in range(n_cutoffs):
        filtered_y, filtered_y_preds = filter_array_label_and_predictions_to_array(y_arrays, y_pred_arrays, label_array_filters_for_all_cutoffs, cutoff=i)
        
        
        precisions.append(precision_score(filtered_y, filtered_y_preds))
        recalls.append(recall_score(filtered_y, filtered_y_preds))
        fbetas.append(fbeta_score(filtered_y, filtered_y_preds, beta=beta))
        AUCs.append(roc_auc_score(filtered_y, filtered_y_preds))
        
    PRFAUC_table = pd.DataFrame(index=all_cutoffs)
    PRFAUC_table.index.name = "Cutoffs"
    
    PRFAUC_table["precision"] = precisions
    PRFAUC_table["recall"] = recalls
    PRFAUC_table["F"+str(beta)] = fbetas
    PRFAUC_table["ROC/AUC"] = AUCs
    
    return PRFAUC_table



@njit
def _calculate_bootstrap_stats(y_arrays, y_pred_arrays, label_array_filters_for_all_cutoffs, beta,  bootstrap_iterations=10000):
    
    n_cutoffs = len(label_array_filters_for_all_cutoffs[0])
    n_dfs = len(y_arrays)
    
    PRFAUC_table_per_iteration = np.zeros((n_cutoffs, 4, bootstrap_iterations))
    metric_per_iteration = np.zeros((bootstrap_iterations,))
    
    for i in range(bootstrap_iterations):
        
        bootstrap_indices  = np.random.choice(n_dfs, size=n_dfs)
        
        y_arrays_bootstrapped = [y_arrays[j] for j in bootstrap_indices]
        y_pred_arrays_bootstrapped = [y_pred_arrays[j] for j in bootstrap_indices]
        label_filters_for_all_cutoffs_bootstrapped = [label_array_filters_for_all_cutoffs[j] for j in bootstrap_indices]
        
        
        PRFAUC_table_per_iteration[:,:,i] = calculate_PRFAUC_table(y_arrays_bootstrapped, y_pred_arrays_bootstrapped, label_filters_for_all_cutoffs_bootstrapped, beta)
    #     metric_per_iteration[i] = cutoff_averaged_f_beta(y_arrays_bootstrapped, y_pred_arrays_bootstrapped, label_filters_for_all_cutoffs_bootstrapped, beta)
        
    # metric_mean = np.mean(metric_per_iteration)
    # metric_std = np.std(metric_per_iteration)
    
    # PRFAUC_table_mean = np.mean(PRFAUC_table_per_iteration, axis=2)
    # PRFAUC_table_std = np.std(PRFAUC_table_per_iteration, axis=2)
    
    # return metric_mean, metric_std, PRFAUC_table_mean, PRFAUC_table_std