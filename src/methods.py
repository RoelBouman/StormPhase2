
import pandas as pd
import numpy as np
import ruptures as rpt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import IsolationForest as IF

from .helper_functions import filter_label_and_scores_to_array
from .evaluation import f_beta


class DoubleThresholdMethod:
    
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, interpolation_range_length=10000):
        self.all_cutoffs_ = list(label_filters_for_all_cutoffs[0].keys())
        
        self.evaluation_score_ = {}
        self.precision_ = {}
        self.recall_ = {}
        self.thresholds_ = {}
        
        #safe the two thresholds in a tuple
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
            
            # use only negative values if searching for the lower threshold, only positive if searching for the upper threshold 
            if upper:
                filtered_y = filtered_y[filtered_y_scores >= 0]
                filtered_y_scores = filtered_y_scores[filtered_y_scores >= 0]
            else:
                filtered_y = filtered_y[filtered_y_scores < 0]
                filtered_y_scores = np.abs(filtered_y_scores[filtered_y_scores < 0])
                                            
            self.precision_[str(cutoffs)], self.recall_[str(cutoffs)], self.thresholds_[str(cutoffs)] = precision_recall_curve(filtered_y, filtered_y_scores)
            
            self.evaluation_score_[str(cutoffs)] = score_function(self.precision_[str(cutoffs)], self.recall_[str(cutoffs)])
            
            current_min_threshold = np.min(np.min(self.thresholds_[str(cutoffs)]))
            if current_min_threshold < min_threshold:
                min_threshold = current_min_threshold
                
            current_max_threshold = np.max(np.max(self.thresholds_[str(cutoffs)]))
            if current_max_threshold > max_threshold:
                max_threshold = current_max_threshold
        
        self.interpolation_range_ = np.linspace(max_threshold, min_threshold, interpolation_range_length)
        
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
    

class IsolationForest:
    
    def __init__(self, score_function=f_beta, **params):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.score_function = score_function
        
        # define IsolationForest model
        self.model = IF(**params)
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column

        y_scores_dfs = []
        no_nan_diffs = []
        
        for i, X_df in enumerate(X_dfs):
            # remove all NaN in X, y and label_filters data
            no_nan_diff = X_df['diff_original'].dropna().values.reshape(-1,1)
            y_dfs[i] = y_dfs[i][X_df['diff_original'].notna()]
            
            for key in label_filters_for_all_cutoffs[i].keys():
                label_filters_for_all_cutoffs[i][key] = label_filters_for_all_cutoffs[i][key][X_df['diff_original'].notna()]
                
            no_nan_diffs.append(no_nan_diff)
            
        if fit:
            #flatten helper and fit model on that
            flat_no_nan_diffs = [i for sl in no_nan_diffs for i in sl]
            self.model.fit(flat_no_nan_diffs)
        
        for diff in no_nan_diffs:
            # calculate and scale the scores
            score = self.model.decision_function(diff)
            scaled_score = np.max(score) - (score - 1)
            y_scores_dfs.append(pd.DataFrame(scaled_score))

        if fit:
            self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function)
            
        y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs, self.optimal_threshold_)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=False)

    
class BinarySegmentation:
    
    def __init__(self, score_function=f_beta, **params):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.score_function = score_function
        
        # define IsolationForest model
        self.model = rpt.Binseg(**params)
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column

        y_scores_dfs = []
        
        no_nan_diffs = []
        
        for i, X_df in enumerate(X_dfs):
            # remove all NaN in X, y and label_filters data
            no_nan_diff = X_df['diff_original'].dropna().values.reshape(-1,1)
            y_dfs[i] = y_dfs[i][X_df['diff_original'].notna()]
            
            for key in label_filters_for_all_cutoffs[i].keys():
                label_filters_for_all_cutoffs[i][key] = label_filters_for_all_cutoffs[i][key][X_df['diff_original'].notna()]
                
            no_nan_diffs.append(no_nan_diff)
        
        for i, diff in enumerate(no_nan_diffs):           
            # defining the penalty
            n = len(diff) # nr of samples
            sigma = np.std(diff)
            penalty = np.log(n) * sigma**2 # https://arxiv.org/pdf/1801.00718.pdf
            
            bkps = self.model.fit_predict(diff, pen = penalty)
            
            y_scores_dfs.append(pd.DataFrame(self.data_to_score(diff, bkps)))

        if fit:
            self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function)
            
        y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs, self.optimal_threshold_)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=False)
    
    def data_to_score(self, df, bkps):
        y_score = np.zeros(len(df))
        total_mean = np.mean(df) # calculate mean of all values in timeseries
        
        prev_bkp = 0
                
        for bkp in bkps:
            segment = df[prev_bkp:bkp] # define a segment between two breakpoints
            segment_mean = np.mean(segment)
            
            # for all values in segment, set its score to th difference between the total mean and the mean of the segment its in
            y_score[prev_bkp:bkp] = total_mean - segment_mean   
            
            prev_bkp = bkp
        
        return y_score            
        
        
class SingleThresholdStatisticalProfiling(StatisticalProfiling, SingleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)
        
class DoubleThresholdStatisticalProfiling(StatisticalProfiling, DoubleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)
        
class SingleThresholdIsolationForest(IsolationForest, SingleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)

class SingleThresholdBinarySegmentation(BinarySegmentation, SingleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)
        
class DoubleThresholdBinarySegmentation(BinarySegmentation, SingleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)