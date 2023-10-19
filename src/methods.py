import pandas as pd
import numpy as np
import ruptures as rpt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import IsolationForest as IF

from .helper_functions import filter_label_and_scores_to_array
from .evaluation import f_beta

class ThresholdMethod:
    def _calculate_interpolated_recall_precision(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold, interpolation_range_length=10000):
        
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
    
    def _calculate_interpolated_scores(self, interpolated_recall, interpolated_precision, used_cutoffs, score_function):
        
        interpolated_scores = np.zeros((len(interpolated_recall), len(used_cutoffs)))
        for i, cutoffs in enumerate(used_cutoffs):
            interpolated_scores[:,i] = score_function(interpolated_precision[str(cutoffs)], interpolated_recall[str(cutoffs)])
            
        interpolated_scores = pd.DataFrame(interpolated_scores, columns=[str(cutoffs) for cutoffs in used_cutoffs])
        return interpolated_scores
        
    
    def _find_max_score_index_for_cutoffs(self, scores_over_cutoffs, used_cutoffs):
    
        column_labels = [str(cutoff) for cutoff in used_cutoffs]
        mean_score_over_cutoffs = np.mean(scores_over_cutoffs[column_labels], axis=1)
        
        max_score_index = np.argmax(mean_score_over_cutoffs)
    
        return max_score_index

class DoubleThresholdMethod(ThresholdMethod):
    
    def __init__(self):
        self.scores_calculated = False
        self.is_single_threshold_method = False
    
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, used_cutoffs, recalculate_scores=False, interpolation_range_length=10000):
        self.all_cutoffs = list(label_filters_for_all_cutoffs[0].keys())
        
        if not all([str(used_cutoff) in self.all_cutoffs for used_cutoff in used_cutoffs]):
            raise ValueError("Not all used cutoffs: " +str(used_cutoffs) +" are in all cutoffs used in preprocessing: " + str(self.all_cutoffs))
                
        if not self.scores_calculated or recalculate_scores:
            self.negative_threshold_recall, self.negative_threshold_precision, self.negative_thresholds = self._calculate_interpolated_recall_precision(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="negative")
            self.positive_threshold_recall, self.positive_threshold_precision, self.positive_thresholds = self._calculate_interpolated_recall_precision(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="positive")
            
            self.score_calculated = True

            
        self.negative_scores = self._calculate_interpolated_scores(self.negative_threshold_recall, self.negative_threshold_precision, used_cutoffs, score_function)
        self.positive_scores = self._calculate_interpolated_scores(self.positive_threshold_recall, self.positive_threshold_precision, used_cutoffs, score_function)
        
        negative_max_score_index = self._find_max_score_index_for_cutoffs(self.negative_scores, used_cutoffs)
        positive_max_score_index = self._find_max_score_index_for_cutoffs(self.positive_scores, used_cutoffs)
        
        #calculate optimal thresholds (negative threshold needs to be set to be negative)
        self.optimal_negative_threshold = -self.negative_thresholds[negative_max_score_index]
        self.optimal_positive_threshold = self.positive_thresholds[positive_max_score_index]
            
    def predict_from_scores_dfs(self, y_scores_dfs):
        
        y_prediction_dfs = []
        for score in y_scores_dfs:
            pred = np.zeros((score.shape[0],))
            pred[np.logical_or(np.squeeze(score) < self.optimal_negative_threshold, np.squeeze(score) >= self.optimal_positive_threshold)] = 1
            y_prediction_dfs.append(pd.Series(pred).to_frame(name="label"))
            
        return y_prediction_dfs


class SingleThresholdMethod(ThresholdMethod):
    #score function must accept precision and recall as input
    #score function should be maximized
    def __init__(self):
        self.scores_calculated = False
        self.is_single_threshold_method = True
        
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, used_cutoffs, recalculate_scores=False, interpolation_range_length=10000):
        self.all_cutoffs = list(label_filters_for_all_cutoffs[0].keys())
        
        if not all([str(used_cutoff) in self.all_cutoffs for used_cutoff in used_cutoffs]):
            raise ValueError("Not all used cutoffs: " +str(used_cutoffs) +" are in all cutoffs used in preprocessing: " + str(self.all_cutoffs))
                
        if not self.scores_calculated or recalculate_scores:
            self.recall, self.precision, self.thresholds = self._calculate_interpolated_recall_precision(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="symmetrical")
            
            self.score_calculated = True
            
        self.scores = self._calculate_interpolated_scores(self.recall, self.precision, used_cutoffs, score_function)
        
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


class StatisticalProfiling:
    
    def __init__(self, score_function=f_beta, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], quantiles=(10,90), ):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.quantiles = quantiles
        self.score_function = score_function
        self.used_cutoffs = used_cutoffs
    
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        y_scores_dfs = []
        
        for X_df in X_dfs:
            scaler = RobustScaler(quantile_range=self.quantiles)
            y_scores_dfs.append(pd.DataFrame(scaler.fit_transform(X_df["diff"].values.reshape(-1,1))))
            
        if fit:
            self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function, self.used_cutoffs)
            
        y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=False)
    

class IsolationForest:
    
    def __init__(self, score_function=f_beta, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], **params):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.score_function = score_function
        self.used_cutoffs = used_cutoffs
        
        # define IsolationForest model
        self.model = IF(**params)
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column

        y_scores_dfs = []
            
        if fit:
            #flatten and fit model on that
            flat_X_dfs_diff = np.array([i for X_df in X_dfs for i in X_df['diff'].values.reshape(-1,1)])
            self.model.fit(flat_X_dfs_diff)
            
        scores = []
        station_maxs = []
        for X_df in X_dfs:
            scores.append(self.model.decision_function(X_df['diff'].values.reshape(-1,1)))
            station_maxs.append(np.max(scores[-1]))
        max_score = max(station_maxs)
            
        for score in scores:
            # calculate and scale the scores
            scaled_score = max_score - (score - 1)
            y_scores_dfs.append(pd.DataFrame(scaled_score))

        if fit:
            self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function, self.used_cutoffs)
            
        y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=False)

    
class BinarySegmentation:
    
    def __init__(self, score_function=f_beta, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], beta=0.12, quantiles=(10,90), penalty="fused_lasso", scaling=True, **params):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.score_function = score_function
        self.beta = beta
        self.quantiles = quantiles
        self.scaling = scaling
        self.penalty = penalty        
        self.used_cutoffs = used_cutoffs
        
        # define Binseg model
        self.model = rpt.Binseg(**params)
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column

        y_scores_dfs = []
        
        if self.scaling:
            scaler = RobustScaler(quantile_range=self.quantiles)
        
        for i, X_df in enumerate(X_dfs): 
            signal = X_df['diff'].values.reshape(-1,1)
            
            if self.scaling:
                signal = scaler.fit_transform(signal)
            
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
            
            y_scores_dfs.append(pd.DataFrame(self.data_to_score(signal, bkps)))

        if fit:
            self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function, self.used_cutoffs)
            
        y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=False)
    
    def fused_lasso_penalty(self, signal, beta):
        tot_sum = 0
        mean = np.mean(signal)
        for i in signal:
            tot_sum += np.abs(i - mean)
        
        return beta * tot_sum
    
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
        SingleThresholdMethod.__init__(self)
        
class DoubleThresholdStatisticalProfiling(StatisticalProfiling, DoubleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)
        DoubleThresholdMethod.__init__(self)
        
class SingleThresholdIsolationForest(IsolationForest, SingleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)
        SingleThresholdMethod.__init__(self)

class SingleThresholdBinarySegmentation(BinarySegmentation, SingleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)
        SingleThresholdMethod.__init__(self)
        
class DoubleThresholdBinarySegmentation(BinarySegmentation, DoubleThresholdMethod):
    
    def __init__(self, **params):
        super().__init__(**params)
        DoubleThresholdMethod.__init__(self)
        
    
class StackEnsemble:
    
    def __init__(self, method_classes, method_hyperparameter_dict_list, cutoffs_per_method, score_function=f_beta):
        
        self.__is_ensemble = True
        
        self.method_classes = method_classes
        self.method_hyperparameter_dicts = method_hyperparameter_dict_list
        self.cutoffs_per_method = cutoffs_per_method
        self.score_function = f_beta
        
        self.models = [method(**hyperparameters, used_cutoffs=used_cutoffs) for method, hyperparameters, used_cutoffs in zip(method_classes, method_hyperparameter_dict_list, self.cutoffs_per_method)]
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=True):
        self._scores = []
        self._predictions = []
        for model in self.models:
            scores, predictions = model.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, fit)
            self._scores.append(scores)
            self._predictions.append(predictions)
            
        return(self._scores, self._combine_predictions(self._predictions))
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs):
        self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, fit=False)
        
    def _combine_predictions(self, prediction_list):
        combined_predictions = []
        n_stations = len(prediction_list[0])
        for i in range(n_stations):
            station_i_prediction_dfs = []
            for prediction_dfs in prediction_list:
                station_i_prediction_dfs.append(prediction_dfs[i])
            combined_predictions.append(pd.DataFrame(np.logical_or.reduce(station_i_prediction_dfs), columns=["label"]))
        return combined_predictions
    
class NaiveStackEnsemble(StackEnsemble):
    def __init__(self, method_classes, method_hyperparameter_dict_list, all_cutoffs, score_function=f_beta):
        cutoffs_per_method = [all_cutoffs]*len(method_classes)
        
        super().__init__(method_classes, method_hyperparameter_dict_list, cutoffs_per_method, score_function=f_beta)