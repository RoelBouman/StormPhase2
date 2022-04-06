#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""
import numpy as np
from src.evaluation import double_threshold_and_score
from src.evaluation import threshold_and_score

from src.evaluation import inverse_threshold_and_score

#uses grid search
def find_BS_thresholds(y_scores, y_true, lengths, cutoffs):
    unique_scores = np.unique(y_scores)
    
    thresholds = (unique_scores[:-1] + unique_scores[1:])/2
    
    
    best_score = 0
    for lower_threshold in thresholds:
        upper_thresholds = (x for x in thresholds if x > lower_threshold)
        for upper_threshold in upper_thresholds:
            
            score = double_threshold_and_score((lower_threshold, upper_threshold), y_true, y_scores, lengths, cutoffs)
            
            if score > best_score:
                best_score = score
                optimal_thresholds = (lower_threshold, upper_threshold)
    
    return optimal_thresholds


def find_BS_thresholds4(y_scores, y_true, lengths, cutoffs):
    unique_scores = np.unique(y_scores)
    
    thresholds = (unique_scores[:-1] + unique_scores[1:])/2
    
    
    best_score = 0
    for lower_threshold in thresholds[:len(thresholds)//2]:
        
        for upper_threshold in thresholds[len(thresholds)//2:]:
            
            score = double_threshold_and_score((lower_threshold, upper_threshold), y_true, y_scores, lengths, cutoffs)

            if score > best_score:
                best_score = score
                optimal_thresholds = (lower_threshold, upper_threshold)
    
    return optimal_thresholds


def score_per_threshold(y_scores, y_true, lengths, cutoffs):
    unique_scores = np.unique(y_scores)
    
    thresholds = (unique_scores[:-1] + unique_scores[1:])/2
    
    lower_thresholds = thresholds[:len(thresholds)//2]
    upper_thresholds = thresholds[len(thresholds)//2:]
    
    best_score = 0
    
    score_grid = np.zeros((len(lower_thresholds), len(upper_thresholds)))
    
    for i, lower_threshold in enumerate(lower_thresholds):
        
        for j, upper_threshold in enumerate(upper_thresholds):
            
            score = double_threshold_and_score((lower_threshold, upper_threshold), y_true, y_scores, lengths, cutoffs)
            print((lower_threshold, upper_threshold))
            print(score)
            score_grid[i,j] = score
            if score > best_score:
                best_score = score
                optimal_thresholds = (lower_threshold, upper_threshold)
    
    return (score_grid, lower_thresholds, upper_thresholds)

#uses single + double cutoff method for fewer passes
def find_BS_thresholds2(y_scores, y_true, lengths, cutoffs):
    unique_scores = np.unique(y_scores)
    
    thresholds = (unique_scores[:-1] + unique_scores[1:])/2
    
    lower_thresholds = thresholds[:len(thresholds)//2]
    upper_thresholds = thresholds[len(thresholds)//2:]
    
    best_score = 0
    
    for upper_threshold in upper_thresholds:
        
        score = double_threshold_and_score((np.min(thresholds), upper_threshold), y_true, y_scores, lengths, cutoffs)

        print(score)
        if score > best_score:
            best_score = score
            best_upper_threshold = upper_threshold
            
            
    print("lower thresholds:")
    print("TESTTESTTESTTESTTEST")
    for lower_threshold in lower_thresholds:
            
        score = double_threshold_and_score((lower_threshold, best_upper_threshold), y_true, y_scores, lengths, cutoffs)
        print(score)
        if score > best_score:
            best_score = score
            best_lower_threshold = lower_threshold
    
    
    return (best_lower_threshold, best_upper_threshold)

#uses single + double cutoff method for fewer passes, redoes initial upper_threshold guess
def find_BS_thresholds5(y_scores, y_true, lengths, cutoffs):
    unique_scores = np.unique(y_scores)
    
    thresholds = (unique_scores[:-1] + unique_scores[1:])/2
    
    lower_thresholds = thresholds[:len(thresholds)//2]
    upper_thresholds = thresholds[len(thresholds)//2:]
    
    best_score = 0
    
    for upper_threshold in upper_thresholds:
        score = double_threshold_and_score((np.min(thresholds), upper_threshold), y_true, y_scores, lengths, cutoffs)

        print(score)
        if score > best_score:
            best_score = score
            best_upper_threshold = upper_threshold
            
            
    print("lower thresholds:")
    for lower_threshold in lower_thresholds:
            
        score = double_threshold_and_score((lower_threshold, best_upper_threshold), y_true, y_scores, lengths, cutoffs)
        print(score)
        if score > best_score:
            best_score = score
            best_lower_threshold = lower_threshold
            
    print("second upper thresholds pass")
    for upper_threshold in upper_thresholds:
        print(upper_threshold)
        
        score = double_threshold_and_score((best_lower_threshold, upper_threshold), y_true, y_scores, lengths, cutoffs)

        
        print(score)
        if score > best_score:
            best_score = score
            best_upper_threshold = upper_threshold
    
    
    return (best_lower_threshold, best_upper_threshold)


#uses single threshold
def find_BS_thresholds3(y_scores, y_true, lengths, cutoffs):
    y_scores = np.abs(y_scores)
    unique_scores = np.unique(y_scores)
    
    thresholds = (unique_scores[:-1] + unique_scores[1:])/2
    
    best_score = 0
    
    for upper_threshold in thresholds:
        print(upper_threshold)
        
        score = inverse_threshold_and_score(upper_threshold, y_true, y_scores, lengths, cutoffs)
        #score_test = double_threshold_and_score
        
        print(score)
        if score > best_score:
            best_score = score
            best_upper_threshold = upper_threshold
            
    return best_upper_threshold