#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:02:32 2023

@author: rbouman
"""
import numpy as np
import pandas as pd
from numba import njit, prange

from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from .helper_functions import filter_label_and_predictions_to_array, filter_label_and_predictions

def f_beta(precision, recall, beta=1, eps=np.finfo(float).eps):
    
    return(1+beta**2) * (precision*recall)/((beta**2*precision)+recall+eps)

@njit
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
    X_mins_no_filter, X_maxs_no_filter = [], []
    
    for X_df, y_df, y_pred_df in zip(X_dfs, y_dfs, y_pred_dfs):
        X_normal = X_df[y_df["label"]==0][load_column]
        X_pred_normal = X_df[y_pred_df["label"]==0][load_column]

        X_mins.append(X_normal.min())
        X_maxs.append(X_normal.max())

        #When entire station is filtered, manually set min and max to 0:
        X_pred_min = X_pred_normal.min()
        if np.isnan(X_pred_min):
            X_pred_min = 0
            
        X_pred_max = X_pred_normal.max()
        if np.isnan(X_pred_max):
            X_pred_max = 0
            
        X_pred_mins.append(X_pred_min)
        X_pred_maxs.append(X_pred_max)
        
        X_mins_no_filter.append(X_df[load_column].min())
        X_maxs_no_filter.append(X_df[load_column].max())
        
    return X_mins, X_maxs, X_pred_mins, X_pred_maxs, X_mins_no_filter, X_maxs_no_filter


def calculate_signed_and_relative_stats(X_dfs, y_dfs, y_pred_dfs, load_column="S_original", eps=np.finfo(float).eps):
    X_mins, X_maxs, X_pred_mins, X_pred_maxs, X_mins_no_filter, X_maxs_no_filter = calculate_minmax_stats(X_dfs, y_dfs, y_pred_dfs, load_column)
    
    min_differences = [X_min - X_pred_min for X_min, X_pred_min in zip(X_mins, X_pred_mins)]
    max_differences = [X_max - X_pred_max for X_max, X_pred_max in zip(X_maxs, X_pred_maxs)]
    
    has_negative_load = [X_min < 0 for X_min in X_mins]
    
    relative_min_differences = [(min_difference)/(X_min+eps) for X_min, min_difference in zip(X_mins, min_differences)]
    relative_max_differences = [(max_difference)/(X_max+eps) for X_max, max_difference in zip(X_maxs, max_differences)]
    
    minmax_stats = min_differences, max_differences, relative_min_differences, relative_max_differences, X_mins, X_pred_mins, X_maxs, X_pred_maxs, has_negative_load, X_mins_no_filter, X_maxs_no_filter
    stats_df = pd.DataFrame(minmax_stats).T
    stats_df.columns = "min_differences", "max_differences", "relative_min_differences", "relative_max_differences", "X_mins", "X_pred_mins", "X_maxs", "X_pred_maxs", "has_negative_load", "X_mins_no_filter", "X_maxs_no_filter"
    
    return stats_df

@njit(parallel=True)
def bootstrap_stats_per_confmat_array(bootstrap_samples, confmat_per_station, beta, eps=np.finfo(float).eps):
    
    bootstrap_iterations = bootstrap_samples.shape[0]
    
    metrics = np.zeros((bootstrap_iterations, 3), dtype=np.float64)
    #in order: precision, recall, Fbeta
    
    for i in prange(bootstrap_iterations):
        indices = bootstrap_samples[i,:]
        
        confmats = [confmat_per_station[k] for k in indices]
        
        confmat_sum = np.array([0,0,0,0])
        for confmat in confmats:
            confmat_sum += confmat
            
        tn, fp, fn, tp = confmat_sum
        
        metrics[i,0] = tp/(tp+fp)
        metrics[i,1] = tp/(tp+fn)
        metrics[i,2] = (1+beta**2) * (tp) / ((1+beta**2)*tp + (beta**2)*fn + fp + eps)
        
    mean_metrics = np.zeros((3,), dtype=np.float64)
    mean_metrics[0] = np.mean(metrics[:,0])
    mean_metrics[1] = np.mean(metrics[:,1])
    mean_metrics[2] = np.mean(metrics[:,2])
    
    std_metrics = np.zeros((3,), dtype=np.float64)
    std_metrics[0] = np.std(metrics[:,0])
    std_metrics[1] = np.std(metrics[:,1])
    std_metrics[2] = np.std(metrics[:,2])
    
    fbetas = metrics[:,2]
    
    return mean_metrics, std_metrics, fbetas


def calculate_bootstrap_stats(y_dfs, y_pred_dfs, label_filters_for_all_cutoffs, beta, bootstrap_iterations=10000):

    all_cutoffs = label_filters_for_all_cutoffs[0].keys()
    
    n_cutoffs = len(all_cutoffs)
    n_dfs = len(y_dfs)

    #confmat_per_station_per_cutoff = {}
    
    bootstrap_samples = np.random.choice(np.arange(n_dfs, dtype=np.int32), size=(bootstrap_iterations, n_dfs))
    
    table_mean = np.zeros((n_cutoffs,3))
    table_std = np.zeros((n_cutoffs,3))
    fbetas = np.zeros((n_cutoffs, bootstrap_iterations,))
    
    for i, cutoffs in enumerate(all_cutoffs):
        filtered_y, filtered_y_preds = filter_label_and_predictions(y_dfs, y_pred_dfs, label_filters_for_all_cutoffs, cutoffs)
        
        confmat_per_station = np.zeros((len(y_dfs),4), dtype=np.int32)
        for j, (y, y_pred) in enumerate(zip(filtered_y, filtered_y_preds)):
            confmat_per_station[j,:] = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
        
        #confmat_per_station_per_cutoff[str(cutoffs)] = confmat_per_station
        
        table_mean[i,:], table_std[i,:], fbetas[i,:] = bootstrap_stats_per_confmat_array(bootstrap_samples, confmat_per_station, beta)
    
    
    PRF_mean_table = pd.DataFrame(index=all_cutoffs)
    PRF_mean_table.index.name = "Cutoffs"
    
    PRF_mean_table["precision"] = table_mean[:,0]
    PRF_mean_table["recall"] = table_mean[:,1]
    PRF_mean_table["F"+str(beta)] = table_mean[:,2]
    
    PRF_std_table = pd.DataFrame(index=all_cutoffs)
    PRF_std_table.index.name = "Cutoffs"
    
    PRF_std_table["precision"] = table_std[:,0]
    PRF_std_table["recall"] = table_std[:,1]
    PRF_std_table["F"+str(beta)] = table_std[:,2]
    
    avg_fbeta_mean = np.mean(np.mean(fbetas, axis=0))
    avg_fbeta_std = np.std(np.mean(fbetas, axis=0))
    
    #if any nans, repeat procedure:
    if PRF_mean_table.isnull().any().any():
        print("NaN in validation results due to invalid split, recalculating:...")
        PRF_mean_table, PRF_std_table, avg_fbeta_mean, avg_fbeta_std = calculate_bootstrap_stats(y_dfs, y_pred_dfs, label_filters_for_all_cutoffs, beta, bootstrap_iterations)

    return PRF_mean_table, PRF_std_table, avg_fbeta_mean, avg_fbeta_std

