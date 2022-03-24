#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""


import numpy as np

from typing import List, Tuple


def stats_per_cutoff(y_true: np.array, y_pred: np.array, lengths: np.array, lower_cutoff: float, upper_cutoff: float) -> Tuple:
    
    filter_condition = np.logical_or(lengths==0, np.logical_and(lengths > lower_cutoff, lengths <= upper_cutoff))#these samples are included
    
    y_true = y_true[filter_condition]
    y_pred = y_pred[filter_condition]
    
    TN = np.logical_and(y_true == 0, y_pred == 0).sum()
    FP = np.logical_and(y_true == 0, y_pred == 1).sum()
    FN = np.logical_and(y_true == 1, y_pred == 0).sum()
    TP = np.logical_and(y_true == 1, y_pred == 1).sum()
    
    return (TN, FP, FN, TP)

def calculate_stats(y_true: np.array, y_pred: np.array, lengths: np.array, cutoffs: List[Tuple]) -> Tuple:
    
    TN = np.zeros((len(cutoffs)))
    FP = np.zeros((len(cutoffs)))
    FN = np.zeros((len(cutoffs)))
    TP = np.zeros((len(cutoffs)))
    
    # Calculate true_negatives, false_positives, false_negatives and TP per cutoff
    for i, (lower_cutoff, upper_cutoff) in enumerate(cutoffs):
        TN[i], FP[i], FN[i], TP[i] = stats_per_cutoff(y_true, y_pred, lengths, lower_cutoff, upper_cutoff)
        
    return TN, FP, FN, TP

def fbeta_from_confmat(tn, fp, fn, tp, beta=10):
    return (1+beta**2)*tp / ((1+beta**2) * tp + beta**2 * fn + fp)

def _STORM_score(TN, FP, FN, TP, beta=10):
    
    score_part = np.zeros((len(TN)))
    for i, (tn, fp, fn, tp) in enumerate(zip(TN, FP, FN, TP)):
        score_part[i] = fbeta_from_confmat(fn, fp, fn, tp, beta)
        
    return(np.mean(score_part))
    
def STORM_score(y_true: np.array, y_pred: np.array, lengths: np.array, cutoffs: List[Tuple], beta=10) -> float:
    TN, FP, FN, TP = calculate_stats(y_true, y_pred, lengths, cutoffs)
    return _STORM_score(TN, FP, FN, TP, beta)
    
def threshold_scores(y_scores: np.ndarray, threshold:float) -> np.array:
    return (y_scores < threshold).astype(float)

def threshold_and_score(threshold: float, y_true: np.array, y_scores: np.array, lengths: np.array, cutoffs: List[Tuple], beta=10) -> float:
    if type(threshold) is tuple:
        threshold = threshold[0]
    return( STORM_score(y_true, threshold_scores(y_scores, threshold), lengths, cutoffs, beta))
    
def inv_threshold_and_score(threshold: float, y_true: np.array, y_scores: np.array, lengths: np.array, cutoffs: List[Tuple], beta=10) -> float:
    if type(threshold) is tuple:
        threshold = threshold[0]
    return(-STORM_score(y_true, threshold_scores(y_scores, threshold), lengths, cutoffs, beta))
    