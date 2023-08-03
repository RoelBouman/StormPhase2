
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve

from .helper_functions import filter_label_and_scores_to_array
from .evaluation import f_beta

class DoubleThresholdMethod:
    
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, interpolation_range_length=10000):
        self.all_cutoffs_ = list(label_filters_for_all_cutoffs[0].keys())
        
        self.evaluation_score_ = {}
        self.precision_ = {}
        self.recall_ = {}
        self.thresholds_ = {}
         
        self.optimal_threshold_ = (self.optimize_single_threshold(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, upper = True),
                                   self.optimize_single_threshold(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, upper = False))
        
        
    def optimize_single_threshold(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, upper, interpolation_range_length=10000):
        #min and max thresholds are tracked for interpolation across all cutoff categories
        min_threshold = 0
        max_threshold = 0
        
        #for all cutoffs, calculate concatenated labels and scores, filtered
        #calculate pr curve for each cutoffs in all_cutoffs
        #combine pr curves according to score function and objective to find optimum
        for cutoffs in self.all_cutoffs_:
            filtered_y, filtered_y_scores = filter_label_and_scores_to_array(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, cutoffs)
            
            if upper:
                filtered_y_scores = np.array([i for i in filtered_y_scores if i >= 0])
                upper = False
            else:
                filtered_y_scores = np.array([i for i in filtered_y_scores if i < 0])
            
            self.precision_[str(cutoffs)], self.recall_[str(cutoffs)], self.thresholds_[str(cutoffs)] = precision_recall_curve(filtered_y, filtered_y_scores)
            
            self.evaluation_score_[str(cutoffs)] = score_function(self.precision_[str(cutoffs)], self.recall_[str(cutoffs)])
            
            current_min_threshold = np.min(np.min(self.thresholds_[str(cutoffs)]))
            if current_min_threshold < min_threshold:
                min_threshold = current_min_threshold
                
            current_max_threshold = np.max(np.max(self.thresholds_[str(cutoffs)]))
            if current_max_threshold > max_threshold:
                max_threshold = current_max_threshold
        
        self.interpolation_range_ = np.linspace(min_threshold, max_threshold, interpolation_range_length)
        
        self.mean_score_over_cutoffs_ = np.zeros(self.interpolation_range_.shape)
        for cutoffs in self.all_cutoffs_:
             
             self.mean_score_over_cutoffs_ += np.interp(self.interpolation_range_, self.thresholds_[str(cutoffs)], self.evaluation_score_[str(cutoffs)][:-1])
        
        self.mean_score_over_cutoffs_ /= len(self.all_cutoffs_)
        
        max_score_index = np.argmax(self.mean_score_over_cutoffs_)
        
        return self.interpolation_range_[max_score_index]

    def predict_from_scores_dfs(self, y_scores_dfs, thresholds):
        lower_threshold = thresholds[0]
        upper_threshold = thresholds[1]
        
        y_prediction_dfs = []
        for score in y_scores_dfs:
            pred = np.zeros((score.shape[0],))
            pred[np.logical_or(np.squeeze(score) <= lower_threshold, np.squeeze(score) >= upper_threshold)] = 1
            y_prediction_dfs.append(pd.Series(pred).to_frame())
            
        return y_prediction_dfs

class SingleThresholdMethod:
    #score function must accept precision and recall as input
    #score function should be maximized
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, interpolation_range_length=10000):
        self.all_cutoffs_ = list(label_filters_for_all_cutoffs[0].keys())
        
        self.evaluation_score_ = {}
        self.precision_ = {}
        self.recall_ = {}
        self.thresholds_ = {}
        
        #min and max thresholds are tracked for interpolation across all cutoff categories
        min_threshold = 0
        max_threshold = 0
        #for all cutoffs, calculate concatenated labels and scores, filtered
        #calculate pr curve for each cutoffs in all_cutoffs
        #combine pr curves according to score function and objective to find optimum
        for cutoffs in self.all_cutoffs_:
            filtered_y, filtered_y_scores = filter_label_and_scores_to_array(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, cutoffs)
            
            
            filtered_y_scores = np.abs(filtered_y_scores)
            self.precision_[str(cutoffs)], self.recall_[str(cutoffs)], self.thresholds_[str(cutoffs)] = precision_recall_curve(filtered_y, filtered_y_scores)
            
            self.evaluation_score_[str(cutoffs)] = score_function(self.precision_[str(cutoffs)], self.recall_[str(cutoffs)])
            
            current_min_threshold = np.min(np.min(self.thresholds_[str(cutoffs)]))
            if current_min_threshold < min_threshold:
                min_threshold = current_min_threshold
                
            current_max_threshold = np.max(np.max(self.thresholds_[str(cutoffs)]))
            if current_max_threshold > max_threshold:
                max_threshold = current_max_threshold
        
        self.interpolation_range_ = np.linspace(min_threshold, max_threshold, interpolation_range_length)
        
        self.mean_score_over_cutoffs_ = np.zeros(self.interpolation_range_.shape)
        for cutoffs in self.all_cutoffs_:
             
             self.mean_score_over_cutoffs_ += np.interp(self.interpolation_range_, self.thresholds_[str(cutoffs)], self.evaluation_score_[str(cutoffs)][:-1])
        
        self.mean_score_over_cutoffs_ /= len(self.all_cutoffs_)
        
        max_score_index = np.argmax(self.mean_score_over_cutoffs_)
        
        self.optimal_threshold_ = self.interpolation_range_[max_score_index]
        

    def predict_from_scores_dfs(self, y_scores_dfs, threshold):
        y_prediction_dfs = []
        for score in y_scores_dfs:
            pred = np.zeros((score.shape[0],))
            pred[np.abs(np.squeeze(score)) >= threshold] = 1
            y_prediction_dfs.append(pd.Series(pred).to_frame())
            
        return y_prediction_dfs

class StatisticalProfiling:
    
    def __init__(self, score_function=f_beta, quantiles=(10,90)):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.quantiles=quantiles
        self.score_function=score_function
    
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        
        y_scores_dfs = []
        
        for X_df in X_dfs:
            scaler = RobustScaler(quantile_range=self.quantiles)
            y_scores_dfs.append(pd.DataFrame(scaler.fit_transform(X_df["diff"].values.reshape(-1,1))))
            
        if fit:
            self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function)
            
        y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs, self.optimal_threshold_)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=False)
    
        
class SingleThresholdStatisticalProfiling(StatisticalProfiling, SingleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)
        
class DoubleThresholdStatisticalProfiling(StatisticalProfiling, DoubleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)