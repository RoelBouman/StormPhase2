#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:02:32 2023

@author: rbouman
"""
import numpy as np
from sklearn.metrics import fbeta_score

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