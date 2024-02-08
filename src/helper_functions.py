 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""
import numpy as np
# import pandas as pd
# from numba import njit

def filter_dfs_to_array(dfs, df_filters):
    #filter defines what to exclude, so, True is excluded
    filtered_dfs = [df[np.logical_not(df_filter)] for df, df_filter in zip(dfs, df_filters)]
    
    return np.concatenate(filtered_dfs)

def filter_label_and_predictions_to_array(y_dfs, y_preds_dfs, label_filters_for_all_cutoffs, cutoffs):
    df_filters = [filter_df[str(cutoffs)] for filter_df in label_filters_for_all_cutoffs]
    y_label_dfs = [df["label"] for df in y_dfs]
    
    filtered_y_labels = filter_dfs_to_array(y_label_dfs, df_filters)
    filtered_y_preds = filter_dfs_to_array(y_preds_dfs, df_filters)
    
    return filtered_y_labels, filtered_y_preds


def filter_label_and_scores_to_array(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, cutoffs):
    df_filters = [filter_df[str(cutoffs)] for filter_df in label_filters_for_all_cutoffs]
    y_label_dfs = [df["label"] for df in y_dfs]
    
    filtered_y_labels = filter_dfs_to_array(y_label_dfs, df_filters)
    filtered_y_scores = filter_dfs_to_array(y_scores_dfs, df_filters).squeeze()
    
    return filtered_y_labels, filtered_y_scores

# def df_list_to_array_list(df_list, column):
#     array_list = [df[column].to_numpy(dtype=np.int32) for df in df_list]

#     return array_list

# def label_filters_to_array(label_filters_for_all_cutoffs):
#     filters_array_list = [pd.DataFrame(filters).to_numpy(dtype=np.bool_) for filters in label_filters_for_all_cutoffs]
    
#     return filters_array_list

# @njit
# def filter_array_label_and_predictions_to_array(y_arrays, y_pred_arrays, label_array_filters_for_all_cutoffs=list(np.array([[]])), cutoff_index=0):
#     y_arrays_cat = np.concatenate(y_arrays)
#     y_pred_arrays_cat = np.concatenate(y_pred_arrays)
#     label_filter_arrays = np.concatenate(label_array_filters_for_all_cutoffs, axis=0)
        
#     label_filter = label_filter_arrays[:,cutoff_index]
    
#     return y_arrays_cat[label_filter], y_pred_arrays_cat[label_filter]