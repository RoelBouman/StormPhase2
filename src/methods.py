from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler

def produce_predictions_single_threshold(y_score_dfs, threshold):
    y_prediction_dfs = []
    for score in y_score_dfs:
        pred = np.zeros((len(score),))
        pred[score > threshold]
    return y_prediction_dfs

def produce_predictions_double_threshold(y_score_dfs, threshold):
    
    return y_prediction_dfs

def optimize_single_threshold(y_dfs, y_score_dfs, score_function, objective):
    
    return 3#optimal_threshold

def optimize_double_threshold(y_dfs, y_score_dfs, score_function, objective):
    
    return optimal_thresholds


class StatisticalProfiling:
    
    def __init__(self, score_function, quantiles=(10,90), objective="maximize"):
        
        self.quantiles=quantiles
        self.score_function=score_function
        self.objective=objective
    
    def fit_transform(self, X_dfs, y_dfs):
        #X_dfs needs at least "diff column
        
        
        scaler = RobustScaler(quantile_range=self.quantiles)
        
        y_score_dfs = []
        
        for X_df, y_df in zip(X_dfs, y_dfs):
            
            y_score_dfs.append(pd.DataFrame(scaler.fit_transform(X_df["diff"].values.reshape(-1,1))))
            
        #self.optimize_thresholds(y_dfs, y_score_dfs)
        
        #y_prediction_dfs = self.produce_predictions(y_score_dfs)
        
        return y_score_dfs#, y_prediction_dfs
    
    @abstractmethod
    def optimize_thresholds(self, y_dfs, y_score_dfs):
        raise NotImplementedError
        
    @abstractmethod
    def produce_predictions(self, y_score_dfs):
        raise NotImplementedError
        
        
class SingleThresholdStatisticalProfiling(StatisticalProfiling):
    
    def __init__(self, **params):
        super().__init__(**params)
        
    def optimize_thresholds(self, y_dfs, y_score_dfs):
        self.threshold = optimize_single_threshold(y_dfs, y_score_dfs, self.score_function, self.objective)
        
    def produce_predictions(self, y_score_dfs):
        return produce_predictions_single_threshold(y_score_dfs, self.threshold)
    
class DoubleThresholdStatisticalProfiling(StatisticalProfiling):
    
    def __init__(self, **params):
        super().__init__(**params)
        
    def optimize_thresholds(self, y_dfs, y_score_dfs):
        self.threshold = optimize_double_threshold(y_dfs, y_score_dfs, self.score_function, self.objective)
        
    def produce_predictions(self, y_score_dfs):
        return produce_predictions_double_threshold(y_score_dfs, self.threshold)