 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""
import numpy as np

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