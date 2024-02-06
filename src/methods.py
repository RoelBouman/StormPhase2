from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import ruptures as rpt

import os
import pickle
from hashlib import sha256

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.ensemble import IsolationForest as IF

from .helper_functions import filter_label_and_scores_to_array
from .evaluation import f_beta, f_beta_from_confmat

class IndependentDoubleThresholdMethod:
    #score function must accept false_positives, true_positives,false_negatives as input
    #score function should be maximized
    def __init__(self, score_function=None, score_function_kwargs=None):
        self.scores_calculated = False
        
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
        
        self.quantiles = quantiles
        self.used_cutoffs = used_cutoffs
    
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=True, dry_run=False):
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
        
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and not overwrite:
            with open(scores_path, 'rb') as handle:
                y_scores_dfs = pickle.load(handle)
            with open(predictions_path, 'rb') as handle:
                y_prediction_dfs = pickle.load(handle)
            self.load_model()
        else:
            
            y_scores_dfs = []
            
            for X_df in X_dfs:
                scaler = RobustScaler(quantile_range=self.quantiles)
                y_scores_dfs.append(pd.DataFrame(scaler.fit_transform(X_df["diff"].values.reshape(-1,1))))
                
            if fit:
                self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.used_cutoffs)
                
            y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
            
            if not dry_run:
                with open(scores_path, 'wb') as handle:
                    pickle.dump(y_scores_dfs, handle)
                with open(predictions_path, 'wb') as handle:
                    pickle.dump(y_prediction_dfs, handle)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=False)
    
    def get_model_string(self):
        model_string = str({"quantiles":self.quantiles}).encode("utf-8")
        
        return model_string
    

class IsolationForest(ScoreCalculator):
    
    def __init__(self, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], forest_per_station=True, scaling=False, quantiles=(10,90), **params):
        super().__init__()
        # Scaling is only done when forest_per_station = False, quantiles is only used when scaling=True and forest-per_station=False
        
        self.used_cutoffs = used_cutoffs
        self.forest_per_station = forest_per_station
        self.params = params
        self.scaling = scaling
        self.quantiles = quantiles
        
        # define IsolationForest model
        self.model = IF(**params)
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=True, dry_run=False):
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
        
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and not overwrite:
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
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=False)
    
    def get_model_string(self):
        hyperparam_dict = {}
        hyperparam_dict["params"] = self.params
        hyperparam_dict["forest_per_station"] = self.forest_per_station
        hyperparam_dict["scaling"] = self.scaling
        hyperparam_dict["quantiles"] = self.quantiles
        model_string = str(hyperparam_dict).encode("utf-8")
        
        return model_string
    
class BinarySegmentation(ScoreCalculator):
    
    def __init__(self, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], beta=0.12, quantiles=(10,90), penalty="fused_lasso", scaling=True, reference_point="median", **params):
        super().__init__()
        self.beta = beta
        self.quantiles = quantiles
        self.scaling = scaling
        self.penalty = penalty        
        self.used_cutoffs = used_cutoffs
        self.params = params
        self.reference_point = reference_point
        
        # define Binseg model
        self.model = rpt.Binseg(**params)
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=True, dry_run=False):
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
        
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and not overwrite:
            with open(scores_path, 'rb') as handle:
                y_scores_dfs = pickle.load(handle)
            with open(predictions_path, 'rb') as handle:
                y_prediction_dfs = pickle.load(handle)
            self.load_model()
            
            
            
        else:
            y_scores_dfs = []
            
            if self.scaling:
                scaler = RobustScaler(quantile_range=self.quantiles)
            
            for i, X_df in enumerate(X_dfs): 
                signal = X_df["diff"].values.reshape(-1,1)
                
                if self.scaling:
                    signal = scaler.fit_transform(signal)
                
                bkps = self.get_breakpoints(signal)

                y_scores_dfs.append(pd.DataFrame(self.data_to_score(signal, bkps, self.reference_point)))
    
            if fit:
                self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.used_cutoffs)
                
            y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
            
            if not dry_run:
                with open(scores_path, 'wb') as handle:
                    pickle.dump(y_scores_dfs, handle)
                with open(predictions_path, 'wb') as handle:
                    pickle.dump(y_prediction_dfs, handle)
        
        return y_scores_dfs, y_prediction_dfs
    
    def get_breakpoints(self, signal):
        """
        Find and return the breakpoints in a given dataframe

        Parameters
        ----------
        df : dataframe
            the dataframe in which the breakpoints must be found
        
        Returns
        -------
        list of integers
            the integers represent the positions of the breakpoints found, always includes len(df)

        """
        
        # decide the penalty https://arxiv.org/pdf/1801.00718.pdf
        if self.penalty == 'lin':
            n = len(signal)
            penalty = n * self.beta
        elif self.penalty == 'fused_lasso':
            penalty = self.fused_lasso_penalty(signal, self.beta)
        else:
            # if no correct penalty selected, raise exception
            raise Exception("Incorrect penalty")
            
        bkps = self.model.fit_predict(signal, pen = penalty)
        
        return bkps
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=False)
    
    def fused_lasso_penalty(self, signal, beta):
        tot_sum = 0
        mean = np.mean(signal)
        for i in signal:
            tot_sum += np.abs(i - mean)
        
        return beta * tot_sum
    
    def data_to_score(self, df, bkps, reference_point):
        y_score = np.zeros(len(df))
        
        if reference_point.lower() == "mean":
            ref_point = np.mean(df) # calculate mean of all values in timeseries
        elif reference_point.lower() == "median":
            ref_point = np.median(df)
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
                ref_point = np.mean(df[first_bkp_longest_segment:last_bkp_longest_segment])
            elif reference_point.lower() == "longest_median":
                ref_point = np.median(df[first_bkp_longest_segment:last_bkp_longest_segment])
                
            self.reference_point_value = ref_point
        else:
            raise ValueError("reference_point needs to be =: {'median', 'mean', 'longest_mean', 'longest_median'}")
        
        prev_bkp = 0
                
        for bkp in bkps:
            segment = df[prev_bkp:bkp] # define a segment between two breakpoints
            segment_mean = np.mean(segment)
            
            # for all values in segment, set its score to th difference between the total mean and the mean of the segment its in
            y_score[prev_bkp:bkp] = ref_point - segment_mean   
            
            prev_bkp = bkp
        
        return y_score            
        
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
    
    def save_model(self, overwrite=True):
        method_path = os.path.join(self.base_models_path, self.method_name, self.preprocessing_hash)
        os.makedirs(method_path, exist_ok=True)
        full_path = os.path.join(method_path, self.filename)
        
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
        
class IndependentDoubleThresholdStatisticalProcessControl(StatisticalProcessControl, IndependentDoubleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, score_function=None, score_function_kwargs=None, **params):
        super().__init__(**params)
        IndependentDoubleThresholdMethod.__init__(self, score_function=score_function, score_function_kwargs=score_function_kwargs)
        self.method_name = "IndependentDoubleThresholdSPC"
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
        
class IndependentDoubleThresholdBinarySegmentation(BinarySegmentation, IndependentDoubleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, score_function=None, score_function_kwargs=None, **params):
        super().__init__(**params)
        IndependentDoubleThresholdMethod.__init__(self, score_function=score_function, score_function_kwargs=score_function_kwargs)
        self.method_name = "IndependentDoubleThresholdBS"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)
        
class SaveableEnsemble(SaveableModel):
    
    def load_model(self):
        method_path = os.path.join(self.base_models_path, self.method_name, self.preprocessing_hash)
        full_path = os.path.join(method_path, self.filename)
        f = open(full_path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()      
        self.__dict__.update(tmp_dict) 
        
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

        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=True, dry_run=False):
        self._scores = []
        temp_scores = []
        self._predictions = []
        for model in self.models:
            scores, predictions = model.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit, dry_run)
            temp_scores.append(scores)
            self._predictions.append(predictions)
        
        self._scores = [pd.concat([scores[i] for scores in temp_scores], axis=1) for i in range(len(temp_scores[0]))]
        return(self._scores, self._combine_predictions(self._predictions))
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=False)
        
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
    
    def save_model(self, overwrite=True):
        for model in self.models:
            model.save_model(overwrite)
        
        method_path = os.path.join(self.base_models_path, self.method_name, self.preprocessing_hash)
        os.makedirs(method_path, exist_ok=True)
        full_path = os.path.join(method_path, self.filename)
        
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