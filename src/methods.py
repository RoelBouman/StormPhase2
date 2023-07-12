from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve

from .helper_functions import filter_dfs_to_array

def predict_from_scores_single_threshold(y_scores_dfs, threshold):
    y_prediction_dfs = []
    for score in y_scores_dfs:
        pred = np.zeros((score.shape[0],))
        pred[np.squeeze(score) > threshold] = 1
        y_prediction_dfs.append(pd.Series(pred).to_frame())
        
    return y_prediction_dfs

def predict_from_scores_double_threshold(y_scores_dfs, thresholds):
    lower_threshold = thresholds[0]
    upper_threshold = thresholds[1]
    
    y_prediction_dfs = []
    for score in y_scores_dfs:
        pred = np.zeros((score.shape[0],))
        pred[np.squeeze(score) > lower_threshold & np.squeeze(score) < upper_threshold] = 1
        y_prediction_dfs.append(pd.Series(pred).to_frame())
        
    return y_prediction_dfs

def optimize_single_threshold(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, objective):
    all_cutoffs = label_filters_for_all_cutoffs[0].keys()
    
    #for all cutoffs, calculate concatenated labels and scores, filtered
    #calculate det curve for each cutoffs in all_cutoffs
    #combine det curves according to score function and objective to find optimum
    for cutoffs in all_cutoffs:
        df_filters = [filter_df[str(cutoffs)] for filter_df in label_filters_for_all_cutoffs]
        y_label_dfs = [df["label"] for df in y_dfs]
        
        filtered_y = filter_dfs_to_array(y_label_dfs, df_filters)
        filtered_y_scores = filter_dfs_to_array(y_scores_dfs, df_filters).squeeze()
        
        precision, recall, thresholds = precision_recall_curve(filtered_y, filtered_y_scores)
        print(average_precision_score(filtered_y, filtered_y_scores))
        
    return 2#optimal_threshold

def optimize_double_threshold(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, objective):
    
    return (2,2)


# NB: score function must take output of https://scikit-learn.org/stable/modules/generated/sklearn.metrics.det_curve.html
class StatisticalProfiling:
    
    def __init__(self, score_function, quantiles=(10,90), objective="maximize"):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.quantiles=quantiles
        self.score_function=score_function
        self.objective=objective
    
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        
        scaler = RobustScaler(quantile_range=self.quantiles)
        
        y_scores_dfs = []
        
        for X_df in X_dfs:
            
            y_scores_dfs.append(pd.DataFrame(scaler.fit_transform(X_df["diff"].values.reshape(-1,1))))
            
        self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs)
        
        y_prediction_dfs = self.predict_from_scores(y_scores_dfs)
        
        return y_scores_dfs, y_prediction_dfs
    
    @abstractmethod
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs):
        raise NotImplementedError
        
    @abstractmethod
    def predict_from_scores(self, y_scores_dfs):
        raise NotImplementedError
        
        
class SingleThresholdStatisticalProfiling(StatisticalProfiling):
    
    def __init__(self, **params):
        super().__init__(**params)
        
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs):
        self.threshold_ = optimize_single_threshold(label_filters_for_all_cutoffs, y_dfs, y_scores_dfs, self.score_function, self.objective)
        
    def predict_from_scores(self, y_scores_dfs):
        return predict_from_scores_single_threshold(y_scores_dfs, self.threshold_)
    
class DoubleThresholdStatisticalProfiling(StatisticalProfiling):
    
    def __init__(self, **params):
        super().__init__(**params)
        
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs):
        self.threshold_ = optimize_double_threshold(label_filters_for_all_cutoffs, y_dfs, y_scores_dfs, self.score_function, self.objective)
        
    def predict_from_scores(self, y_scores_dfs):
        return predict_from_scores_double_threshold(y_scores_dfs, self.threshold_)