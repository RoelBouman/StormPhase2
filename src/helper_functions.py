#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""
import numpy as np
from src.evaluation import double_threshold_and_score
from src.evaluation import threshold_and_score

from src.evaluation import inverse_threshold_and_score

#uses single + double cutoff method for fewer passes, redoes initial upper_threshold guess
def find_BS_thresholds(y_scores, y_true, lengths, cutoffs):
    unique_scores = np.unique(y_scores)
    
    thresholds = (unique_scores[:-1] + unique_scores[1:])/2
    
    lower_thresholds = thresholds[thresholds < 0]
    upper_thresholds = thresholds[thresholds >= 0]
    
    best_score = 0
    
    for upper_threshold in upper_thresholds:
        score = double_threshold_and_score((np.min(thresholds), upper_threshold), y_true, y_scores, lengths, cutoffs)


        if score > best_score:
            best_score = score
            best_upper_threshold = upper_threshold
            
            

    for lower_threshold in lower_thresholds:
            
        score = double_threshold_and_score((lower_threshold, best_upper_threshold), y_true, y_scores, lengths, cutoffs)

        if score > best_score:
            best_score = score
            best_lower_threshold = lower_threshold
            

    for upper_threshold in upper_thresholds:

        
        score = double_threshold_and_score((best_lower_threshold, upper_threshold), y_true, y_scores, lengths, cutoffs)

        

        if score > best_score:
            best_score = score
            best_upper_threshold = upper_threshold
    
    
    return (best_lower_threshold, best_upper_threshold)