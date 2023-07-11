from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler

def predict_from_scores_single_threshold(y_score_dfs, threshold):
    y_prediction_dfs = []
    for score in y_score_dfs:
        pred = np.zeros((score.shape[0],))
        pred[np.squeeze(score) > threshold] = 1
        y_prediction_dfs.append(pd.Series(pred).to_frame())
        
    return y_prediction_dfs

def predict_from_scores_double_threshold(y_score_dfs, thresholds):
    lower_threshold = thresholds[0]
    upper_threshold = thresholds[1]
    
    y_prediction_dfs = []
    for score in y_score_dfs:
        pred = np.zeros((score.shape[0],))
        pred[np.squeeze(score) > lower_threshold & np.squeeze(score) < upper_threshold] = 1
        y_prediction_dfs.append(pd.Series(pred).to_frame())
        
    return y_prediction_dfs

def optimize_single_threshold(label_filters_for_all_cutoffs, y_score_dfs, score_function, objective):
    
    return 2#optimal_threshold

def optimize_double_threshold(label_filters_for_all_cutoffs, y_score_dfs, score_function, objective):
    
    return (2,2)


class StatisticalProfiling:
    
    def __init__(self, score_function, quantiles=(10,90), objective="maximize"):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.quantiles=quantiles
        self.score_function=score_function
        self.objective=objective
    
    def fit_transform_predict(self, X_dfs, labels_for_all_cutoffs):
        #X_dfs needs at least "diff column
        
        
        scaler = RobustScaler(quantile_range=self.quantiles)
        
        y_score_dfs = []
        
        for X_df, y_df in zip(X_dfs, labels_for_all_cutoffs):
            
            y_score_dfs.append(pd.DataFrame(scaler.fit_transform(X_df["diff"].values.reshape(-1,1))))
            
        self.optimize_thresholds(labels_for_all_cutoffs, y_score_dfs)
        
        y_prediction_dfs = self.predict_from_scores(y_score_dfs)
        
        return y_score_dfs, y_prediction_dfs
    
    @abstractmethod
    def optimize_thresholds(self, labels_for_all_cutoffs, y_score_dfs):
        raise NotImplementedError
        
    @abstractmethod
    def predict_from_scores(self, y_score_dfs):
        raise NotImplementedError
        
        
class SingleThresholdStatisticalProfiling(StatisticalProfiling):
    
    def __init__(self, **params):
        super().__init__(**params)
        
    def optimize_thresholds(self, labels_for_all_cutoffs, y_score_dfs):
        self.threshold_ = optimize_single_threshold(labels_for_all_cutoffs, y_score_dfs, self.score_function, self.objective)
        
    def predict_from_scores(self, y_score_dfs):
        return predict_from_scores_single_threshold(y_score_dfs, self.threshold_)
    
class DoubleThresholdStatisticalProfiling(StatisticalProfiling):
    
    def __init__(self, **params):
        super().__init__(**params)
        
    def optimize_thresholds(self, labels_for_all_cutoffs, y_score_dfs):
        self.threshold_ = optimize_double_threshold(labels_for_all_cutoffs, y_score_dfs, self.score_function, self.objective)
        
    def predict_from_scores(self, y_score_dfs):
        return predict_from_scores_double_threshold(y_score_dfs, self.threshold_)