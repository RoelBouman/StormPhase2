from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import ruptures as rpt

import os
import pickle
from hashlib import sha256

from numba import njit

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.ensemble import IsolationForest as IF

from .helper_functions import filter_label_and_scores_to_array
from .evaluation import f_beta, f_beta_from_confmat

class DoubleThresholdMethod:
    #score function must accept false_positives, true_positives,false_negatives as input
    #score function should be maximized
    def __init__(self, score_function=None, score_function_kwargs=None):
        self.scores_calculated = False
        self.threshold_optimization_method = "DoubleThreshold"
        
        if score_function is None:
            try:
                self.score_function_beta = score_function_kwargs["beta"]
            except KeyError:
                raise KeyError("If score_function is set to None, score_function_kwargs should contain key:value pair for 'beta':..." )
            
            self.score_function_kwargs = score_function_kwargs
            self.score_function = self.score_function_from_confmat_with_beta
        
        else:
            self.score_function = self.custom_score_function_from_confmat
            self.score_function_kwargs = score_function_kwargs
            
    def score_function_from_confmat_with_beta(self, fps, tps, fns, **kwargs):
        return f_beta_from_confmat(fps, tps, fns, **kwargs)
    
    def custom_score_function_from_confmat(self, score_function, *args, **kwargs):
        return score_function(*args, **kwargs)
        
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, used_cutoffs, recalculate_scores=False, interpolation_range_length=1000):
        self.all_cutoffs = list(label_filters_for_all_cutoffs[0].keys())
        
        if not all([str(used_cutoff) in self.all_cutoffs for used_cutoff in used_cutoffs]):
            raise ValueError("Not all used cutoffs: " +str(used_cutoffs) +" are in all cutoffs used in preprocessing: " + str(self.all_cutoffs))
                
        if not self.scores_calculated or recalculate_scores:
            self.lower_false_positives, self.lower_true_positives, self.lower_false_negatives, self.negative_thresholds = self._calculate_interpolated_partial_confmat(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="negative", interpolation_range_length=interpolation_range_length)
            self.upper_false_positives, self.upper_true_positives, self.upper_false_negatives, self.positive_thresholds = self._calculate_interpolated_partial_confmat(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="positive", interpolation_range_length=interpolation_range_length)
            
            self.scores_calculated = True

        self.calculate_and_set_thresholds(used_cutoffs)
        
    def _calculate_interpolated_partial_confmat(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold, interpolation_range_length=1000):
        
        fps_ = {}
        tps_ = {}
        fns_ = {}
        thresholds_ = {}
        
        all_cutoffs = list(label_filters_for_all_cutoffs[0].keys())
        
        min_threshold = 0
        max_threshold = 0
        
        for cutoffs in all_cutoffs:
            filtered_y, filtered_y_scores = filter_label_and_scores_to_array(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, cutoffs)
            
            # use only negative values if searching for the lower (negative) threshold, only positive if searching for the upper (positive) threshold 
            if which_threshold == "positive":
                filtered_y = filtered_y[filtered_y_scores >= 0]
                filtered_y_scores = filtered_y_scores[filtered_y_scores >= 0]
            elif which_threshold == "negative":
                filtered_y = filtered_y[filtered_y_scores < 0]
                filtered_y_scores = np.abs(filtered_y_scores[filtered_y_scores < 0])
            else:
                raise ValueError("which_threshold is set incorrectly. Valid options are: {\"positive\", \"negative\"}.")
                                            
            fps_[str(cutoffs)], tps_[str(cutoffs)], thresholds_[str(cutoffs)] = _binary_clf_curve(filtered_y, filtered_y_scores)
            fns_[str(cutoffs)] = tps_[str(cutoffs)][-1] - tps_[str(cutoffs)]
            
            current_min_threshold = np.min(np.min(thresholds_[str(cutoffs)]))
            if current_min_threshold < min_threshold:
                min_threshold = current_min_threshold
                
            current_max_threshold = np.max(np.max(thresholds_[str(cutoffs)]))
            if current_max_threshold > max_threshold:
                max_threshold = current_max_threshold
        
        interpolation_range_ = np.linspace(max_threshold, min_threshold, interpolation_range_length)
        
        interpolated_fps = np.zeros((len(interpolation_range_), len(all_cutoffs)))
        interpolated_tps = np.zeros((len(interpolation_range_), len(all_cutoffs)))
        interpolated_fns = np.zeros((len(interpolation_range_), len(all_cutoffs)))
        
        for i, cutoffs in enumerate(all_cutoffs):
             
             interpolated_fps[:,i] = np.interp(interpolation_range_, thresholds_[str(cutoffs)][::-1], fps_[str(cutoffs)][::-1])
             interpolated_tps[:,i] = np.interp(interpolation_range_, thresholds_[str(cutoffs)][::-1], tps_[str(cutoffs)][::-1])
             interpolated_fns[:,i] = np.interp(interpolation_range_, thresholds_[str(cutoffs)][::-1], fns_[str(cutoffs)][::-1])
             
        
        interpolated_fps = pd.DataFrame(interpolated_fps, columns=[str(cutoffs) for cutoffs in all_cutoffs])
        interpolated_tps = pd.DataFrame(interpolated_tps, columns=[str(cutoffs) for cutoffs in all_cutoffs])
        interpolated_fns = pd.DataFrame(interpolated_fns, columns=[str(cutoffs) for cutoffs in all_cutoffs])
        
        return interpolated_fps, interpolated_tps, interpolated_fns, interpolation_range_
    
    def calculate_and_set_thresholds(self, used_cutoffs):
        
        self.false_positive_grid = {}
        self.true_positive_grid = {}
        self.false_negative_grid = {}
        
        for cutoffs in used_cutoffs:
            fp_grid_1, fp_grid_2 = np.meshgrid(self.lower_false_positives[str(cutoffs)] , self.upper_false_positives[str(cutoffs)] )
            self.false_positive_grid[str(cutoffs)]  = fp_grid_1 + fp_grid_2
            
            tp_grid_1, tp_grid_2 = np.meshgrid(self.lower_true_positives[str(cutoffs)] , self.upper_true_positives[str(cutoffs)] )
            self.true_positive_grid[str(cutoffs)]  = tp_grid_1 + tp_grid_2
            
            fn_grid_1, fn_grid_2 = np.meshgrid(self.lower_false_negatives[str(cutoffs)] , self.upper_false_negatives[str(cutoffs)] )
            self.false_negative_grid[str(cutoffs)]  = fn_grid_1 + fn_grid_2
            
        self.scores = self._calculate_grid_scores(self.false_positive_grid, self.true_positive_grid, self.false_negative_grid, used_cutoffs)
        
        max_score_indices = self._find_max_score_indices_for_cutoffs(self.scores, used_cutoffs)
        
        #calculate optimal thresholds (negative threshold needs to be set to be negative)
        self.optimal_negative_threshold = -self.negative_thresholds[max_score_indices[1]]
        self.optimal_positive_threshold = self.positive_thresholds[max_score_indices[0]]
        
        #Calculate for compatibility later:
        self.optimal_threshold = (self.optimal_negative_threshold, self.optimal_positive_threshold)
        
    def _calculate_grid_scores(self, false_positive_grid, true_positive_grid, false_negative_grid, used_cutoffs):
        
        grid_scores = {}
        for i, cutoffs in enumerate(used_cutoffs):
            grid_scores[str(cutoffs)] = self.score_function(false_positive_grid[str(cutoffs)], true_positive_grid[str(cutoffs)], false_negative_grid[str(cutoffs)], **self.score_function_kwargs)
            
        return grid_scores
        
    
    def _find_max_score_indices_for_cutoffs(self, scores_over_cutoffs, used_cutoffs):
    
        sum_scores = np.zeros(scores_over_cutoffs[str(used_cutoffs[0])].shape)
        
        for cutoffs in used_cutoffs:
            sum_scores += scores_over_cutoffs[str(cutoffs)]
            
        flat_index = np.argmax(sum_scores)
        
        max_score_indices = np.unravel_index(flat_index, sum_scores.shape)
        
        return max_score_indices


    def predict_from_scores_dfs(self, y_scores_dfs):
        
        y_prediction_dfs = []
        for score in y_scores_dfs:
            pred = np.zeros((score.shape[0],))
            pred[np.logical_or(np.squeeze(score) < self.optimal_negative_threshold, np.squeeze(score) >= self.optimal_positive_threshold)] = 1
            y_prediction_dfs.append(pd.Series(pred).to_frame(name="label"))
            
        return y_prediction_dfs
    
    def report_thresholds(self):
        print("Optimal thresholds:")
        print((self.optimal_negative_threshold, self.optimal_positive_threshold))



class SingleThresholdMethod:
    #score function must accept precision and recall as input
    #score function should be maximized
    def __init__(self, score_function = None, score_function_kwargs=None):
        self.scores_calculated = False
        self.threshold_optimization_method = "SingleThreshold"
        
        if score_function is None:
            try:
                self.score_function_beta = score_function_kwargs["beta"]
            except TypeError:
                raise TypeError("If score_function is set to None, score_function_kwargs should contain key:value pair for 'beta':..." )
            
            self.score_function_kwargs = score_function_kwargs
            self.score_function = self.score_function_from_precision_recall_with_beta
        
        else:
            self.score_function = self.custom_score_function_from_precision_recall
            self.score_function_kwargs = score_function_kwargs
        
    def score_function_from_precision_recall_with_beta(self, precision, recall, **kwargs):
        return f_beta(precision, recall, **kwargs)
    
    def custom_score_function_from_precision_recall(self, score_function, *args):
        return score_function(*args, **self.score_function_kwargs)
        
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, used_cutoffs, recalculate_scores=False, interpolation_range_length=1000):
        self.all_cutoffs = list(label_filters_for_all_cutoffs[0].keys())
        
        if not all([str(used_cutoff) in self.all_cutoffs for used_cutoff in used_cutoffs]):
            raise ValueError("Not all used cutoffs: " +str(used_cutoffs) +" are in all cutoffs used in preprocessing: " + str(self.all_cutoffs))
                
        if not self.scores_calculated or recalculate_scores:
            self.recall, self.precision, self.thresholds = self._calculate_interpolated_recall_precision(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="symmetrical", interpolation_range_length=interpolation_range_length)
            
            self.scores_calculated = True
        
        self.calculate_and_set_thresholds(used_cutoffs)
        
    def calculate_and_set_thresholds(self, used_cutoffs):
        self.scores = self._calculate_interpolated_scores(self.recall, self.precision, used_cutoffs)
        
        max_score_index = self._find_max_score_index_for_cutoffs(self.scores, used_cutoffs)
        
        #calculate optimal thresholds (negative threshold needs to be set to be negative)
        self.optimal_threshold = self.thresholds[max_score_index]
        
    def predict_from_scores_dfs(self, y_scores_dfs):
        y_prediction_dfs = []
        for score in y_scores_dfs:
            pred = np.zeros((score.shape[0],))
            pred[np.abs(np.squeeze(score)) >= self.optimal_threshold] = 1
            y_prediction_dfs.append(pd.Series(pred).to_frame(name="label"))
            
        return y_prediction_dfs
    
    def report_thresholds(self):
        print("Optimal threshold:")
        print((self.optimal_threshold))
        
    def _calculate_interpolated_recall_precision(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold, interpolation_range_length=1000):
        
        precision_ = {}
        recall_ = {}
        thresholds_ = {}
        
        all_cutoffs = list(label_filters_for_all_cutoffs[0].keys())
        
        min_threshold = 0
        max_threshold = 0
        
        for cutoffs in all_cutoffs:
            filtered_y, filtered_y_scores = filter_label_and_scores_to_array(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, cutoffs)
            
            # use only negative values if searching for the lower (negative) threshold, only positive if searching for the upper (positive) threshold 
            if which_threshold == "positive":
                filtered_y = filtered_y[filtered_y_scores >= 0]
                filtered_y_scores = filtered_y_scores[filtered_y_scores >= 0]
            elif which_threshold == "negative":
                filtered_y = filtered_y[filtered_y_scores < 0]
                filtered_y_scores = np.abs(filtered_y_scores[filtered_y_scores < 0])
            elif which_threshold == "symmetrical":
                #No additional filtering is needed, but y_scores need to be made absolute
                filtered_y_scores = np.abs(filtered_y_scores)
            else:
                raise ValueError("which_threshold is set incorrectly. Valid options are: {\"positive\", \"negative\", \"symmetrical\"}.")
                                            
            precision_[str(cutoffs)], recall_[str(cutoffs)], thresholds_[str(cutoffs)] = precision_recall_curve(filtered_y, filtered_y_scores)
            
            current_min_threshold = np.min(np.min(thresholds_[str(cutoffs)]))
            if current_min_threshold < min_threshold:
                min_threshold = current_min_threshold
                
            current_max_threshold = np.max(np.max(thresholds_[str(cutoffs)]))
            if current_max_threshold > max_threshold:
                max_threshold = current_max_threshold
        
        interpolation_range_ = np.linspace(max_threshold, min_threshold, interpolation_range_length)
        
        interpolated_recall = np.zeros((len(interpolation_range_), len(all_cutoffs)))
        interpolated_precision = np.zeros((len(interpolation_range_), len(all_cutoffs)))
        for i, cutoffs in enumerate(all_cutoffs):
             
             interpolated_recall[:,i] = np.interp(interpolation_range_, thresholds_[str(cutoffs)], recall_[str(cutoffs)][:-1])
             interpolated_precision[:,i] = np.interp(interpolation_range_, thresholds_[str(cutoffs)], precision_[str(cutoffs)][:-1])
        
        interpolated_recall = pd.DataFrame(interpolated_recall, columns=[str(cutoffs) for cutoffs in all_cutoffs])
        interpolated_precision = pd.DataFrame(interpolated_precision, columns=[str(cutoffs) for cutoffs in all_cutoffs])
        
        return interpolated_recall, interpolated_precision, interpolation_range_
    
    def _calculate_interpolated_scores(self, interpolated_recall, interpolated_precision, used_cutoffs):
        
        interpolated_scores = np.zeros((len(interpolated_recall), len(used_cutoffs)))
        for i, cutoffs in enumerate(used_cutoffs):
            interpolated_scores[:,i] = self.score_function(interpolated_precision[str(cutoffs)], interpolated_recall[str(cutoffs)], **self.score_function_kwargs)
            
        interpolated_scores = pd.DataFrame(interpolated_scores, columns=[str(cutoffs) for cutoffs in used_cutoffs])
        return interpolated_scores
        
    
    def _find_max_score_index_for_cutoffs(self, scores_over_cutoffs, used_cutoffs):
    
        column_labels = [str(cutoff) for cutoff in used_cutoffs]
        mean_score_over_cutoffs = np.mean(scores_over_cutoffs[column_labels], axis=1)
        
        max_score_index = np.argmax(mean_score_over_cutoffs)
    
        return max_score_index

class ScoreCalculator:
    def __init__(self):
        pass
    
    def check_cutoffs(self, cutoffs):
        return cutoffs == self.used_cutoffs
        
class StatisticalProcessControl(ScoreCalculator):
    
    def __init__(self, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], quantiles=(10,90)):
        super().__init__()
        
        self.score_calculation_method_name = "StatisticalProcessControl"
        
        self.quantiles = quantiles
        self.used_cutoffs = used_cutoffs
    
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit=True, dry_run=False, verbose=False):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        #Get paths
        model_name = self.method_name
        score_calculator_name = self.score_calculation_method_name
        hyperparameter_hash = self.get_hyperparameter_hash()        
        
        scores_folder = os.path.join(base_scores_path, score_calculator_name, hyperparameter_hash)
        predictions_folder = os.path.join(base_predictions_path, model_name, hyperparameter_hash)
        
        if not dry_run:
            os.makedirs(scores_folder, exist_ok=True)
            os.makedirs(predictions_folder, exist_ok=True)
        
        scores_path = os.path.join(scores_folder, "scores.pickle")
        predictions_path = os.path.join(predictions_folder, str(self.used_cutoffs)+ ".pickle")
        
        #Calculate scores, or reload if already calculated
        if os.path.exists(scores_path) and not overwrite:
            if verbose:
                print("Scores already exist, reloading")
            with open(scores_path, 'rb') as handle:
                y_scores_dfs = pickle.load(handle)
        else:
            y_scores_dfs = []
            
            for X_df in X_dfs:
                scaler = RobustScaler(quantile_range=self.quantiles)
                y_scores_dfs.append(pd.DataFrame(scaler.fit_transform(X_df["diff"].values.reshape(-1,1))))
        
            if not dry_run:
                with open(scores_path, 'wb') as handle:
                    pickle.dump(y_scores_dfs, handle)
        
        #Calculate predictions from scores
        if os.path.exists(predictions_path) and os.path.exists(self.get_full_model_path()) and not overwrite:
            if verbose:
                print("Predictions and model already exist, reloading")
            with open(predictions_path, 'rb') as handle:
                y_prediction_dfs = pickle.load(handle)
            self.load_model()
        else:
            if fit:
                self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.used_cutoffs)
                
            y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
            
            if not dry_run:
                with open(predictions_path, 'wb') as handle:
                    pickle.dump(y_prediction_dfs, handle)
                if fit:
                    self.save_model()
                    
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, verbose=False):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit=False, verbose=verbose)
    
    def get_model_string(self):
        model_string = str({"quantiles":self.quantiles}).encode("utf-8")
        
        return model_string
    

class IsolationForest(ScoreCalculator):
    
    def __init__(self, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], forest_per_station=True, scaling=False, quantiles=(10,90), **params):
        super().__init__()
        # Scaling is only done when forest_per_station = False, quantiles is only used when scaling=True and forest-per_station=False
        
        self.score_calculation_method_name = "IsolationForest"
        
        self.used_cutoffs = used_cutoffs
        self.forest_per_station = forest_per_station
        self.params = params
        self.scaling = scaling
        self.quantiles = quantiles
        
        # define IsolationForest model
        self.model = IF(**params)
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit=True, dry_run=False, verbose=False):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        model_name = self.method_name
        hyperparameter_hash = self.get_hyperparameter_hash()
        
        scores_path = os.path.join(base_scores_path, model_name, hyperparameter_hash)
        predictions_path = os.path.join(base_predictions_path, model_name, hyperparameter_hash)
        if not dry_run:
            os.makedirs(scores_path, exist_ok=True)
            os.makedirs(predictions_path, exist_ok=True)
        
        scores_path = os.path.join(scores_path, str(self.used_cutoffs)+ ".pickle")
        predictions_path = os.path.join(predictions_path, str(self.used_cutoffs)+ ".pickle")
        
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and os.path.exists(self.get_full_model_path()) and not overwrite:
            if verbose:
                print("Scores/predictions/model already exist, reloading")
            with open(scores_path, 'rb') as handle:
                y_scores_dfs = pickle.load(handle)
            with open(predictions_path, 'rb') as handle:
                y_prediction_dfs = pickle.load(handle)
            self.load_model()
        else:
            y_scores_dfs = []
                
            if not self.forest_per_station and fit:
                    
                #flatten and fit model on that (model is only reused if forest_per_station=False)
                if self.scaling:
                    scaler = RobustScaler(quantile_range=self.quantiles)
                    flat_X_dfs_diff = np.concatenate([scaler.fit_transform(X_df["diff"].values.reshape(-1,1)) for X_df in X_dfs])
                else:
                    flat_X_dfs_diff = np.concatenate([X_df["diff"] for X_df in X_dfs]).reshape(-1,1)
                self.model.fit(flat_X_dfs_diff)
                
            for X_df in X_dfs:
                scaled_score = self.get_IF_scores(X_df)
                y_scores_dfs.append(pd.DataFrame(scaled_score))

            if fit:
                self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.used_cutoffs)
                
            y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
            
            if not dry_run:
                with open(scores_path, 'wb') as handle:
                    pickle.dump(y_scores_dfs, handle)
                with open(predictions_path, 'wb') as handle:
                    pickle.dump(y_prediction_dfs, handle)
                if fit:
                    self.save_model()
        
        return y_scores_dfs, y_prediction_dfs
    
    def get_IF_scores(self, X_df):
        if self.forest_per_station:
            self.model.fit(X_df['diff'].values.reshape(-1,1))
        if not self.scaling:
            score = self.model.decision_function(X_df['diff'].values.reshape(-1,1))
        else:
            scaler = RobustScaler(quantile_range=self.quantiles)
            score = self.model.decision_function(scaler.fit_transform(X_df['diff'].values.reshape(-1,1)))
        scaled_score = -score + 1
        if min(scaled_score) < 0:
            raise ValueError("IF scaled_score lower than 0, something went wrong.")
            
        return scaled_score
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, verbose=False):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit=False, verbose=verbose)
    
    def get_model_string(self):
        hyperparam_dict = {}
        hyperparam_dict["params"] = self.params
        hyperparam_dict["forest_per_station"] = self.forest_per_station
        hyperparam_dict["scaling"] = self.scaling
        hyperparam_dict["quantiles"] = self.quantiles
        model_string = str(hyperparam_dict).encode("utf-8")
        
        return model_string

@njit
def _data_to_score(signal, bkps, reference_point_value):
    y_score = np.zeros(len(signal), dtype=np.float64)
    prev_bkp = 0
    segment_means = []
            
    for bkp in bkps:
        segment = signal[prev_bkp:bkp] # define a segment between two breakpoints
        segment_mean = np.mean(segment)
        
        segment_means.append(segment_mean)
        
        # for all values in segment, set its score to th difference between the total mean and the mean of the segment its in
        y_score[prev_bkp:bkp] = reference_point_value - segment_mean   
        
        prev_bkp = bkp
    
    return y_score, segment_means

class BinarySegmentationBreakpointCalculator():
    
    def __init__(self, beta=0.12, quantiles=(10,90), penalty="L1", scaling=True, reference_point="median", **params):
        self.beta = beta
        self.quantiles = quantiles
        self.scaling = scaling
        self.penalty = penalty
        self.params = params
        self.reference_point = reference_point
        
        # define Binseg model
        self.model = rpt.Binseg(**params)
    
    def get_breakpoints(self, signal):
        """
        Find and return the breakpoints in a given dataframe

        Parameters
        ----------
        signal : dataframe
            the dataframe in which the breakpoints must be found
        
        Returns
        -------
        list of integers
            the integers represent the positions of the breakpoints found, always includes len(signal)

        """
        
        # decide the penalty https://arxiv.org/pdf/1801.00718.pdf
        if self.penalty == 'lin':
            n = len(signal)
            penalty = n * self.beta
        elif self.penalty == 'L1':
            penalty = self.fused_lasso_penalty(signal, self.beta)
        else:
            # if no correct penalty selected, raise exception
            raise Exception("Incorrect penalty")
            
        bkps = self.model.fit_predict(signal, pen = penalty)
        
        return bkps
        
    def fused_lasso_penalty(self, signal, beta):
        mean = np.mean(signal)
        tot_sum = np.sum(np.abs(signal - mean))
        
        return beta * tot_sum
    
    
    def calculate_reference_point_value(self, signal, bkps, reference_point):
        
        if reference_point.lower() == "mean":
            ref_point = np.mean(signal) # calculate mean of all values in timeseries
        elif reference_point.lower() == "median":
            ref_point = np.median(signal)
        elif reference_point.lower() == "longest_mean" or reference_point.lower() == "longest_median": #compare to longest segment mean
            prev_bkp = 0
            longest_segment_length = 0
            for bkp in bkps:
                segment_length = bkp-prev_bkp
                if segment_length > longest_segment_length:
                    first_bkp_longest_segment = prev_bkp
                    last_bkp_longest_segment = bkp
                    longest_segment_length = segment_length
                prev_bkp = bkp
            if reference_point.lower() == "longest_mean":
                ref_point = np.mean(signal[first_bkp_longest_segment:last_bkp_longest_segment])
            elif reference_point.lower() == "longest_median":
                ref_point = np.median(signal[first_bkp_longest_segment:last_bkp_longest_segment])
                
        else:
            raise ValueError("reference_point needs to be =: {'median', 'mean', 'longest_mean', 'longest_median'}")
            
            
        return ref_point
    
    def data_to_score(self, signal, bkps, reference_point):
        
        self.reference_point_value = self.calculate_reference_point_value(signal, bkps, reference_point)
        
        return _data_to_score(signal, bkps, self.reference_point_value)
    
    def get_breakpoints_string(self):
        hyperparam_dict = {}
        hyperparam_dict["beta"] = self.beta
        hyperparam_dict["quantiles"] = self.quantiles
        hyperparam_dict["scaling"] = self.scaling
        hyperparam_dict["penalty"] = self.penalty
        hyperparam_dict["params"] = self.params
        model_string = str(hyperparam_dict).encode("utf-8")
        
        return model_string
    
    def get_breakpoints_hash(self):
        breakpoints_string = self.get_breakpoints_string()
        
        #hash model_string as it can surpass character limit. This also circumvents illegal characters in pathnames for certains OSes
        breakpoints_hash = sha256(breakpoints_string).hexdigest()
        
        return breakpoints_hash


class BinarySegmentation(ScoreCalculator, BinarySegmentationBreakpointCalculator):
    
    def __init__(self, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], beta=0.12, quantiles=(10,90), penalty="L1", scaling=True, reference_point="median", **params):
        super().__init__()
        BinarySegmentationBreakpointCalculator.__init__(self, beta=beta, quantiles=quantiles, penalty=penalty, scaling=scaling, reference_point=reference_point, **params)
        
        self.score_calculation_method_name = "BinarySegmentation"
        
        self.used_cutoffs = used_cutoffs

        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite=False, fit=True, dry_run=False, verbose=False):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        
        #Get paths:
        model_name = self.method_name
        score_calculator_name = self.score_calculation_method_name
        hyperparameter_hash = self.get_hyperparameter_hash()
        breakpoints_hash = self.get_breakpoints_hash()
        
        scores_folder = os.path.join(base_scores_path, score_calculator_name, hyperparameter_hash)
        predictions_folder = os.path.join(base_predictions_path, model_name, hyperparameter_hash)
        breakpoints_folder = os.path.join(base_intermediates_path, score_calculator_name, breakpoints_hash)
        segment_means_folder = os.path.join(base_intermediates_path, score_calculator_name, breakpoints_hash)
        
        if not dry_run:
            os.makedirs(scores_folder, exist_ok=True)
            os.makedirs(predictions_folder, exist_ok=True)
            os.makedirs(breakpoints_folder, exist_ok=True)
        
        scores_path = os.path.join(scores_folder, "scores.pickle")
        predictions_path = os.path.join(predictions_folder, str(self.used_cutoffs)+ ".pickle")
        breakpoints_path = os.path.join(breakpoints_folder, "breakpoints.pickle")
        segment_means_path = os.path.join(segment_means_folder, "segment_means.pickle")
        
        #Get scores
        if os.path.exists(scores_path) and os.path.exists(breakpoints_path) and os.path.exists(segment_means_path) and not overwrite:
            
            if verbose:
                print("Scores already exist, reloading")
            
            with open(scores_path, 'rb') as handle:
                y_scores_dfs = pickle.load(handle)
            with open(segment_means_path, 'rb') as handle:
                self.segment_means_per_station = pickle.load(handle)
            with open(breakpoints_path, 'rb') as handle:
                self.breakpoints_per_station = pickle.load(handle)
                
        else:
            if self.scaling:
                scaler = RobustScaler(quantile_range=self.quantiles)
            
            #calculate signals by getting diff column and optionally scaling
            signals = []
            for X_df in X_dfs: 
                signal = X_df["diff"].values.reshape(-1,1)
                
                if self.scaling:
                    signal = scaler.fit_transform(signal).astype(np.float64).squeeze()
                signals.append(signal)
                
            #load breakpoints if they exist, otherwise calculate them
            if os.path.exists(breakpoints_path) and not overwrite:
                
                if verbose:
                    print("Breakpoints already exist, reloading")
                    
                with open(breakpoints_path, 'rb') as handle:
                    self.breakpoints_per_station = pickle.load(handle)
            else:
                self.breakpoints_per_station = []
                for signal in signals:
                    self.breakpoints_per_station.append(self.get_breakpoints(signal))
                
                if not dry_run:
                    with open(breakpoints_path, 'wb') as handle:
                        pickle.dump(self.breakpoints_per_station, handle)
            
            #Finally: calculate scores
            y_scores_dfs = []
            self.segment_means_per_station = []
            for bkps, signal in zip(self.breakpoints_per_station, signals):
                scores, segment_means = self.data_to_score(signal, bkps, self.reference_point)
                y_scores_dfs.append(pd.DataFrame(scores))
                self.segment_means_per_station.append(segment_means)
            
            if not dry_run:
                with open(scores_path, 'wb') as handle:
                    pickle.dump(y_scores_dfs, handle)
                with open(segment_means_path, 'wb') as handle:
                    pickle.dump(self.segment_means_per_station, handle)
                
            
        #Get predictions from scores, load model if it exists, if not recalculate
        if os.path.exists(predictions_path) and os.path.exists(self.get_full_model_path()) and not overwrite:
            if verbose:
                print("Model and predictions already exist, reloading")
                
            with open(predictions_path, 'rb') as handle:
                y_prediction_dfs = pickle.load(handle)
            self.load_model()
            
        else:
            
            if fit:
                self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.used_cutoffs)
                
            y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
            
            if not dry_run:
                with open(predictions_path, 'wb') as handle:
                    pickle.dump(y_prediction_dfs, handle)
                if fit:
                    self.save_model()
                    
        return y_scores_dfs, y_prediction_dfs
    
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, verbose):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit=False, verbose=verbose)
    
    
    def get_model_string(self):
        hyperparam_dict = {}
        hyperparam_dict["beta"] = self.beta
        hyperparam_dict["quantiles"] = self.quantiles
        hyperparam_dict["scaling"] = self.scaling
        hyperparam_dict["penalty"] = self.penalty
        hyperparam_dict["params"] = self.params
        hyperparam_dict["reference_point"] = self.reference_point
        model_string = str(hyperparam_dict).encode("utf-8")
        
        return model_string
    

        
class SaveableModel(ABC):
    
    def __init__(self, base_models_path, preprocessing_hash):
        self.base_models_path = base_models_path
        self.preprocessing_hash = preprocessing_hash
        self.filename = self.get_filename()
        
        method_path = os.path.join(self.base_models_path, self.method_name, preprocessing_hash)
        full_path = os.path.join(method_path, self.filename)
        
        if os.path.exists(full_path):
            self.load_model()
    
    @abstractmethod
    def get_model_string(self):
        pass
    
    # @property
    # @abstractmethod
    # def method_name(self):
    #     pass

    def get_hyperparameter_hash(self):
        model_string = self.get_model_string()
        
        #hash model_string as it can surpass character limit. This also circumvents illegal characters in pathnames for certains OSes
        hyperparameter_hash = sha256(model_string).hexdigest()
        
        return hyperparameter_hash

    def get_filename(self):
        
        filename = self.get_hyperparameter_hash() + ".pickle"
            
        return filename
    
    def get_full_model_path(self):
        method_path = os.path.join(self.base_models_path, self.method_name, self.preprocessing_hash)
        os.makedirs(method_path, exist_ok=True)
        full_path = os.path.join(method_path, self.filename)
        
        return full_path
    
    def save_model(self, overwrite=True):
        full_path = self.get_full_model_path()
        
        if not os.path.exists(full_path) or overwrite:
            f = open(full_path, 'wb')
            pickle.dump(self.__dict__, f, 2)
            f.close()
        
    def load_model(self):
        
        #manually ensure that used_cutoffs is not overwritten:
        used_cutoffs = self.used_cutoffs
        
        method_path = os.path.join(self.base_models_path, self.method_name, self.preprocessing_hash)
        full_path = os.path.join(method_path, self.filename)
        f = open(full_path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        
        self.__dict__.update(tmp_dict) 
        
        self.used_cutoffs = used_cutoffs
        


class SingleThresholdStatisticalProcessControl(StatisticalProcessControl, SingleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, score_function=None, score_function_kwargs=None, **params):
        super().__init__(**params)
        SingleThresholdMethod.__init__(self, score_function=score_function, score_function_kwargs=score_function_kwargs)
        self.method_name = "SingleThresholdSPC"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)
        
class DoubleThresholdStatisticalProcessControl(StatisticalProcessControl, DoubleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, score_function=None, score_function_kwargs=None, **params):
        super().__init__(**params)
        DoubleThresholdMethod.__init__(self, score_function=score_function, score_function_kwargs=score_function_kwargs)
        self.method_name = "DoubleThresholdSPC"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)
        
class SingleThresholdIsolationForest(IsolationForest, SingleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, score_function=None, score_function_kwargs=None, **params):
        super().__init__(**params)
        SingleThresholdMethod.__init__(self, score_function=score_function, score_function_kwargs=score_function_kwargs)
        self.method_name = "SingleThresholdIF"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)

class SingleThresholdBinarySegmentation(BinarySegmentation, SingleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, score_function=None, score_function_kwargs=None, **params):
        super().__init__(**params)
        SingleThresholdMethod.__init__(self, score_function=score_function, score_function_kwargs=score_function_kwargs)
        self.method_name = "SingleThresholdBS"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)
        
class DoubleThresholdBinarySegmentation(BinarySegmentation, DoubleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, score_function=None, score_function_kwargs=None, **params):
        super().__init__(**params)
        DoubleThresholdMethod.__init__(self, score_function=score_function, score_function_kwargs=score_function_kwargs)
        self.method_name = "DoubleThresholdBS"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)
        
class SaveableEnsemble(SaveableModel):
    
    def get_full_model_path(self):
        method_path = os.path.join(self.base_models_path, self.method_name, self.preprocessing_hash)
        os.makedirs(method_path, exist_ok=True)
        full_path = os.path.join(method_path, self.filename)
        return full_path
    
    def load_model(self):
        full_path = self.get_full_model_path()
        f = open(full_path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()      
        self.__dict__.update(tmp_dict) 
        
        
        
def single_threshold_function(value, threshold):
    return np.abs(value) >= threshold
    
def double_threshold_function(value, thresholds):
    negative_threshold = thresholds[0]
    positive_threshold = thresholds[1]
    return np.logical_or(value < negative_threshold, value >= positive_threshold)

class SequentialEnsemble(SaveableEnsemble):
    
    def __init__(self, base_models_path, preprocessing_hash, segmentation_method, anomaly_detection_method, method_hyperparameter_dict_list, cutoffs_per_method):

        self.is_ensemble = True
        
        self.method_hyperparameter_list = method_hyperparameter_dict_list
        self.cutoffs_per_method = cutoffs_per_method
        self.preprocessing_hash = preprocessing_hash
        
        self.segmentation_method = segmentation_method(base_models_path, preprocessing_hash, **method_hyperparameter_dict_list[0], used_cutoffs=cutoffs_per_method[0])
        self.anomaly_detection_method = anomaly_detection_method(base_models_path, preprocessing_hash, **method_hyperparameter_dict_list[1], used_cutoffs=cutoffs_per_method[1])
        #self.models = [method(base_models_path, preprocessing_hash, **hyperparameters, used_cutoffs=used_cutoffs) for method, hyperparameters, used_cutoffs in zip(method_classes, method_hyperparameter_dict_list, self.cutoffs_per_method)]
        
        self.method_name = "Sequential-"+self.segmentation_method.method_name+"+"+self.anomaly_detection_method.method_name
        
        if self.segmentation_method.threshold_optimization_method == "DoubleThreshold":
            self.threshold_function = double_threshold_function
        elif self.segmentation_method.threshold_optimization_method == "SingleThreshold":
            self.threshold_function = single_threshold_function
        else:
            raise ValueError("Segmentation method is not a {'DoubleThreshold', 'SingleThreshold'} method")
            
        super().__init__(base_models_path, preprocessing_hash)
    
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite=False, fit=True, dry_run=False, verbose=False):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        
        #Get paths:
        model_name = self.method_name
        hyperparameter_hash = self.get_hyperparameter_hash()
        
        scores_folder = os.path.join(base_scores_path, model_name, hyperparameter_hash)
        predictions_folder = os.path.join(base_predictions_path, model_name, hyperparameter_hash)
        
        if not dry_run:
            os.makedirs(scores_folder, exist_ok=True)
            os.makedirs(predictions_folder, exist_ok=True)
        
        scores_path = os.path.join(scores_folder, "scores.pickle")
        predictions_path = os.path.join(predictions_folder, "predictions.pickle")
        
        #Get scores
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and os.path.exists(self.get_full_model_path()) and not overwrite:
            
            if verbose:
                print("Scores/Prediction/Model already exist, reloading")
            
            with open(scores_path, 'rb') as handle:
                final_scores = pickle.load(handle)
            with open(predictions_path, 'rb') as handle:
                final_predictions = pickle.load(handle)
            self.load_model()
        
        else:
            #First fit segmenter based on used_cutoffs for segmentation_method:
            y_scores_dfs_segmenter, y_prediction_dfs_segmenter = self.segmentation_method.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit, dry_run, verbose)
            
            #After initial fit, find segments which are not predicted as 1
            segments_to_anomaly_detector_indices = []
            signal_segments_to_anomaly_detector = []
            label_segments_to_anomaly_detector = []
            label_filter_segments_to_anomaly_detector = []
            
            for X_df, y_df, label_filters, y_prediction_df, segment_means, breakpoints in zip(X_dfs, y_dfs, label_filters_for_all_cutoffs, y_prediction_dfs_segmenter, self.segmentation_method.segment_means_per_station, self.segmentation_method.breakpoints_per_station):
                prev_bkp = 0
                anomalous_segment_indices = []
                anomalous_segment_signal = []
                anomalous_segment_labels = []
                anomalous_segment_label_filters = []
                
                signal = X_df["diff"].to_frame()
                labels = y_df["label"].to_frame()
                
                for segment_mean, bkp in zip(segment_means, breakpoints):
                    #Save segment to later pas to anomaly detection method:
                    if not self.threshold_function(segment_mean, self.segmentation_method.optimal_threshold):
                        signal_segment = signal[prev_bkp:bkp] # define a segment between two breakpoints
                        label_segment = labels[prev_bkp:bkp]                    
                        label_filter_segment = {k:v[prev_bkp:bkp] for k,v in label_filters.items()}
                        
                        anomalous_segment_indices.append((prev_bkp,bkp))
                        anomalous_segment_signal.append(signal_segment)
                        anomalous_segment_labels.append(label_segment)
                        anomalous_segment_label_filters.append(label_filter_segment)
                    
                    prev_bkp = bkp
                    
                signal_segments_to_anomaly_detector.append(anomalous_segment_signal)
                label_segments_to_anomaly_detector.append(anomalous_segment_labels)
                segments_to_anomaly_detector_indices.append(anomalous_segment_indices)
                label_filter_segments_to_anomaly_detector.append(anomalous_segment_label_filters)
            # for each of these segments, apply anomaly detection method to get scores
            # Optimize thresholds for scores of these segments based on used cutoffs for anomaly detection method
            # Obtain predictions per segment
            
            #Before passing to AD method, flatten list of list of dfs to list of dfs
            #We need to keep track of the original station the df belongs to, so we can properly reassign predicted labels later on
            subsignal_df_index = [[i]*len(sublist) for i, sublist in enumerate(signal_segments_to_anomaly_detector)]
            
            subsignal_df_index = [segment for segment_list in subsignal_df_index for segment in segment_list]
            signal_segments_to_anomaly_detector = [segment for segment_list in signal_segments_to_anomaly_detector for segment in segment_list]
            label_segments_to_anomaly_detector = [segment for segment_list in label_segments_to_anomaly_detector for segment in segment_list]
            segments_to_anomaly_detector_indices = [segment for segment_list in segments_to_anomaly_detector_indices for segment in segment_list]
            label_filter_segments_to_anomaly_detector = [segment for segment_list in label_filter_segments_to_anomaly_detector for segment in segment_list]
            
            #Prediction paths don't -need- to be set like this. Most importantly: AD calculation is always unique, as every input breakpoint set is assumed to be unique, so both dry_run and overwrite are manually set to True (possibly overwrite is not 100% needed)
            AD_base_scores_path, AD_base_predictions_path, AD_base_intermediates_path = os.path.join(base_scores_path, "Sequential_AD_part"), os.path.join(base_predictions_path, "Sequential_AD_part"), os.path.join(base_intermediates_path, "Sequential_AD_part")
            ad_scores, ad_predictions = self.anomaly_detection_method.fit_transform_predict(signal_segments_to_anomaly_detector, label_segments_to_anomaly_detector, label_filter_segments_to_anomaly_detector, AD_base_scores_path, AD_base_predictions_path, AD_base_intermediates_path, overwrite=True, fit=fit, dry_run=True, verbose=verbose)
            
            #Recombine predictions of segmenter with predictions of AD method in order to get final predictions
            final_predictions = y_prediction_dfs_segmenter
            
            final_ad_scores = []
            #initialize dfs 
            for score_df in y_scores_dfs_segmenter:
                temp_df = score_df.copy(deep=True)
                temp_df.iloc[:] = np.nan
                final_ad_scores.append(temp_df)
            
            for df_index, ad_score, ad_prediction, segment_indices in zip(subsignal_df_index, ad_scores, ad_predictions, segments_to_anomaly_detector_indices):
                final_predictions[df_index].iloc[segment_indices[0]:segment_indices[1]] = ad_prediction
                final_ad_scores[df_index].iloc[segment_indices[0]:segment_indices[1]] = ad_score
            
            final_scores = [pd.concat([segmenter_score.squeeze(), ad_score.squeeze()], axis=1, keys=[self.segmentation_method.method_name, self.anomaly_detection_method.method_name]) for segmenter_score, ad_score in zip(y_scores_dfs_segmenter, final_ad_scores)]        #Scores should be list of matrices/dfs, with each column indicating the method used for production of said scores
            
            if not dry_run:
                with open(scores_path, 'wb') as handle:
                    pickle.dump(final_scores, handle)
                with open(predictions_path, 'wb') as handle:
                    pickle.dump(final_predictions, handle)
                self.save_model()
            
        return final_scores, final_predictions
        
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, verbose=False):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit=False, verbose=verbose)
        
    def get_model_string(self):
        
        model_string = str(self.method_hyperparameter_list).encode("utf-8")
        
        return model_string
    
    def save_model(self, overwrite=True):
        #for model in self.models:
        #    model.save_model(overwrite)
        
        method_path = os.path.join(self.base_models_path, self.method_name, self.preprocessing_hash)
        os.makedirs(method_path, exist_ok=True)
        full_path = os.path.join(method_path, self.filename)
        
        if not os.path.exists(full_path) or overwrite:
            f = open(full_path, 'wb')
            pickle.dump(self.__dict__, f, 2)
            f.close()
            
    
    def report_thresholds(self):
        models = [self.segmentation_method, self.anomaly_detection_method]
        for model in models:
            print(model.method_name)
            model.report_thresholds()
    
class StackEnsemble(SaveableEnsemble):
    
    def __init__(self, base_models_path, preprocessing_hash, method_classes, method_hyperparameter_dict_list, cutoffs_per_method):

        self.is_ensemble = True
        
        self.method_classes = method_classes
        self.method_hyperparameter_dicts = method_hyperparameter_dict_list
        self.cutoffs_per_method = cutoffs_per_method
        self.preprocessing_hash = preprocessing_hash
        
        self.models = [method(base_models_path, preprocessing_hash, **hyperparameters, used_cutoffs=used_cutoffs) for method, hyperparameters, used_cutoffs in zip(method_classes, method_hyperparameter_dict_list, self.cutoffs_per_method)]
        
        self.method_name = "+".join([model.method_name for model in self.models])
        #self.method_name = "StackEnsemble"
        
        super().__init__(base_models_path, preprocessing_hash)


    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite=False, fit=True, dry_run=False, verbose=False):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        
        #Get paths:
        model_name = self.method_name
        hyperparameter_hash = self.get_hyperparameter_hash()
        
        scores_folder = os.path.join(base_scores_path, model_name, hyperparameter_hash)
        predictions_folder = os.path.join(base_predictions_path, model_name, hyperparameter_hash)
        
        if not dry_run:
            os.makedirs(scores_folder, exist_ok=True)
            os.makedirs(predictions_folder, exist_ok=True)
        
        scores_path = os.path.join(scores_folder, "scores.pickle")
        predictions_path = os.path.join(predictions_folder, "predictions.pickle")
        
        #Get scores
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and os.path.exists(self.get_full_model_path()) and not overwrite:
            
            if verbose:
                print("Scores/Prediction/Model already exist, reloading")
            
            with open(scores_path, 'rb') as handle:
                scores = pickle.load(handle)
            with open(predictions_path, 'rb') as handle:
                predictions = pickle.load(handle)
            self.load_model()
        
        else:
            
            _scores = []
            _predictions = []
            for model in self.models:
                temp_scores, temp_predictions = model.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit, dry_run, verbose)
                _scores.append(temp_scores)
                _predictions.append(temp_predictions)
            
            scores = [pd.concat([scores[i] for scores in _scores], axis=1) for i in range(len(_scores[0]))]
            predictions = self._combine_predictions(_predictions)
    
            
            if not dry_run:
                with open(scores_path, 'wb') as handle:
                    pickle.dump(scores, handle)
                with open(predictions_path, 'wb') as handle:
                    pickle.dump(predictions, handle)
                self.save_model()
            
        return scores, predictions
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, verbose=False):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, base_intermediates_path, overwrite, fit=False, verbose=verbose)
        
    def _combine_predictions(self, prediction_list):
        combined_predictions = []
        n_stations = len(prediction_list[0])
        for i in range(n_stations):
            station_i_prediction_dfs = []
            for prediction_dfs in prediction_list:
                station_i_prediction_dfs.append(prediction_dfs[i])
            combined_predictions.append(pd.DataFrame(np.logical_or.reduce(station_i_prediction_dfs), columns=["label"]))
        return combined_predictions
    
    def get_model_string(self):
        
        model_string = str(self.method_hyperparameter_dicts).encode("utf-8")
        
        return model_string
    
    def get_full_model_path(self):
        method_path = os.path.join(self.base_models_path, self.method_name, self.preprocessing_hash)
        os.makedirs(method_path, exist_ok=True)
        full_path = os.path.join(method_path, self.filename)
        return full_path
    
    def save_model(self, overwrite=True):
        
        full_path = self.get_full_model_path()
        
        if not os.path.exists(full_path) or overwrite:
            f = open(full_path, 'wb')
            pickle.dump(self.__dict__, f, 2)
            f.close()
        
    def report_thresholds(self):
        for model in self.models:
            print(model.method_name)
            model.report_thresholds()
    
class NaiveStackEnsemble(StackEnsemble):
    def __init__(self, base_models_path, preprocessing_hash, method_classes, method_hyperparameter_dict_list, all_cutoffs):
        cutoffs_per_method = [all_cutoffs]*len(method_classes)
        
        super().__init__(base_models_path, preprocessing_hash, method_classes, method_hyperparameter_dict_list, cutoffs_per_method)
        
        self.method_name = "Naive-" + self.method_name