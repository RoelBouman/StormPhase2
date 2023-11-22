from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import ruptures as rpt

import os
import pickle
from hashlib import sha256

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
        #self.is_single_threshold_method = False
    
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, used_cutoffs, recalculate_scores=False, interpolation_range_length=10000):
        self.all_cutoffs = list(label_filters_for_all_cutoffs[0].keys())
        
        if not all([str(used_cutoff) in self.all_cutoffs for used_cutoff in used_cutoffs]):
            raise ValueError("Not all used cutoffs: " +str(used_cutoffs) +" are in all cutoffs used in preprocessing: " + str(self.all_cutoffs))
                
        if not self.scores_calculated or recalculate_scores:
            self.negative_threshold_recall, self.negative_threshold_precision, self.negative_thresholds = self._calculate_interpolated_recall_precision(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="negative")
            self.positive_threshold_recall, self.positive_threshold_precision, self.positive_thresholds = self._calculate_interpolated_recall_precision(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="positive")
            
            self.scores_calculated = True

        self.calculate_and_set_thresholds(used_cutoffs, score_function)
        
    def calculate_and_set_thresholds(self, used_cutoffs, score_function):
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
    
    def report_thresholds(self):
        print("Optimal thresholds:")
        print((self.optimal_negative_threshold, self.optimal_positive_threshold))
        
    def scale_thresholds(self, scaler):
        scaled_optimal_negative_threshold = scaler.inverse_transform(np.array([self.optimal_negative_threshold]).reshape(-1,1))[0][0]
        scaled_optimal_positive_threshold = scaler.inverse_transform(np.array([self.optimal_positive_threshold]).reshape(-1,1))[0][0]
        self.scaled_optimal_threshold = (scaled_optimal_negative_threshold, scaled_optimal_positive_threshold)

class SingleThresholdMethod(ThresholdMethod):
    #score function must accept precision and recall as input
    #score function should be maximized
    def __init__(self):
        self.scores_calculated = False
        #self.is_single_threshold_method = True
        
    def optimize_thresholds(self, y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, score_function, used_cutoffs, recalculate_scores=False, interpolation_range_length=10000):
        self.all_cutoffs = list(label_filters_for_all_cutoffs[0].keys())
        
        if not all([str(used_cutoff) in self.all_cutoffs for used_cutoff in used_cutoffs]):
            raise ValueError("Not all used cutoffs: " +str(used_cutoffs) +" are in all cutoffs used in preprocessing: " + str(self.all_cutoffs))
                
        if not self.scores_calculated or recalculate_scores:
            self.recall, self.precision, self.thresholds = self._calculate_interpolated_recall_precision(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, which_threshold="symmetrical")
            
            self.score_calculated = True
        
        self.calculate_and_set_thresholds(used_cutoffs, score_function)
        
    def calculate_and_set_thresholds(self, used_cutoffs, score_function):
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
    
    def report_thresholds(self):
        print("Optimal threshold:")
        print((self.optimal_threshold))
        
    def scale_thresholds(self, scaler):
        self.scaled_optimal_threshold = scaler.inverse_transform(np.array([self.optimal_threshold]).reshape(-1,1))[0][0]

class ScoreCalculator:
    def __init__(self):
        pass
    
    def check_cutoffs(self, cutoffs):
        return cutoffs == self.used_cutoffs
        
class StatisticalProfiling(ScoreCalculator):
    
    def __init__(self, score_function=f_beta, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], quantiles=(10,90)):
        super().__init__()
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.quantiles = quantiles
        self.score_function = score_function
        self.used_cutoffs = used_cutoffs
    
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        #         pickle.dump(model, handle)
        model_name = self.method_name
        hyperparameter_hash = self.get_hyperparameter_hash()
        
        scores_path = os.path.join(base_scores_path, model_name, hyperparameter_hash)
        predictions_path = os.path.join(base_predictions_path, model_name, hyperparameter_hash)
        os.makedirs(scores_path, exist_ok=True)
        os.makedirs(predictions_path, exist_ok=True)
        scores_path = os.path.join(scores_path, str(self.used_cutoffs)+ ".pickle")
        predictions_path = os.path.join(predictions_path, str(self.used_cutoffs)+ ".pickle")
        
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and not overwrite:
            with open(scores_path, 'rb') as handle:
                y_scores_dfs = pickle.load(handle)
            with open(predictions_path, 'rb') as handle:
                y_prediction_dfs = pickle.load(handle)
        else:
            
            y_scores_dfs = []
            
            for X_df in X_dfs:
                scaler = RobustScaler(quantile_range=self.quantiles).fit(X_df["diff"].values.reshape(-1,1))
                y_scores_dfs.append(pd.DataFrame(scaler.transform(X_df["diff"].values.reshape(-1,1))))
                
            if fit:
                self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function, self.used_cutoffs)
                
                # scale thresholds for visualization (currently not useful as scaler is fit on different data)
                self.scale_thresholds(scaler)
                
            y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
            
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
    
    def __init__(self, score_function=f_beta, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], **params):
        super().__init__()
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        
        self.score_function = score_function
        self.used_cutoffs = used_cutoffs
        self.params = params
        
        # define IsolationForest model
        self.model = IF(**params)
        
        # track scaled scores for visualization
        self.y_scores = []
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        model_name = self.method_name
        hyperparameter_hash = self.get_hyperparameter_hash()
        
        scores_path = os.path.join(base_scores_path, model_name, hyperparameter_hash)
        predictions_path = os.path.join(base_predictions_path, model_name, hyperparameter_hash)
        os.makedirs(scores_path, exist_ok=True)
        os.makedirs(predictions_path, exist_ok=True)
        scores_path = os.path.join(scores_path, str(self.used_cutoffs)+ ".pickle")
        predictions_path = os.path.join(predictions_path, str(self.used_cutoffs)+ ".pickle")
        
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and not overwrite:
            with open(scores_path, 'rb') as handle:
                y_scores_dfs = pickle.load(handle)
            with open(predictions_path, 'rb') as handle:
                y_prediction_dfs = pickle.load(handle)
        else:
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
                self.y_scores.append(scaled_score)
                y_scores_dfs.append(pd.DataFrame(scaled_score))
    
            if fit:
                self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function, self.used_cutoffs)
                
            y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
            
            with open(scores_path, 'wb') as handle:
                pickle.dump(y_scores_dfs, handle)
            with open(predictions_path, 'wb') as handle:
                pickle.dump(y_prediction_dfs, handle)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=False)
    
    def get_model_string(self):
        model_string = str(self.params).encode("utf-8")
        
        return model_string
    
class BinarySegmentation(ScoreCalculator):
    
    def __init__(self, score_function=f_beta, used_cutoffs=[(0, 24), (24, 288), (288, 4032), (4032, np.inf)], beta=0.12, quantiles=(10,90), penalty="fused_lasso", scaling=True, **params):
        # score_function must accept results from sklearn.metrics.det_curve (fpr, fnr, thresholds)
        super().__init__()
        self.score_function = score_function
        self.beta = beta
        self.quantiles = quantiles
        self.scaling = scaling
        self.penalty = penalty        
        self.used_cutoffs = used_cutoffs
        self.params = params
        
        # keep track of breakpoints for visualization
        self.breakpoints_list = []
        
        # define Binseg model
        self.model = rpt.Binseg(**params)
        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=True):
        #X_dfs needs at least "diff" column
        #y_dfs needs at least "label" column
        
        model_name = self.method_name
        hyperparameter_hash = self.get_hyperparameter_hash()
        
        scores_path = os.path.join(base_scores_path, model_name, hyperparameter_hash)
        predictions_path = os.path.join(base_predictions_path, model_name, hyperparameter_hash)
        os.makedirs(scores_path, exist_ok=True)
        os.makedirs(predictions_path, exist_ok=True)
        scores_path = os.path.join(scores_path, str(self.used_cutoffs)+ ".pickle")
        predictions_path = os.path.join(predictions_path, str(self.used_cutoffs)+ ".pickle")
        
        if os.path.exists(scores_path) and os.path.exists(predictions_path) and not overwrite:
            with open(scores_path, 'rb') as handle:
                y_scores_dfs = pickle.load(handle)
            with open(predictions_path, 'rb') as handle:
                y_prediction_dfs = pickle.load(handle)
        else:
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
                self.breakpoints_list.append(bkps)
                
                y_scores_dfs.append(pd.DataFrame(self.data_to_score(signal, bkps)))
    
            if fit:
                self.optimize_thresholds(y_dfs, y_scores_dfs, label_filters_for_all_cutoffs, self.score_function, self.used_cutoffs)
                
                # scale thresholds for visualization
                if self.scaling:
                    self.scale_thresholds(scaler)
                
            y_prediction_dfs = self.predict_from_scores_dfs(y_scores_dfs)
            
            with open(scores_path, 'wb') as handle:
                pickle.dump(y_scores_dfs, handle)
            with open(predictions_path, 'wb') as handle:
                pickle.dump(y_prediction_dfs, handle)
        
        return y_scores_dfs, y_prediction_dfs
    
    def transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite):
        
        return self.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=False)
    
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
        
    def get_model_string(self):
        hyperparam_dict = {}
        hyperparam_dict["beta"] = self.beta
        hyperparam_dict["quantiles"] = self.quantiles
        hyperparam_dict["scaling"] = self.scaling
        hyperparam_dict["penalty"] = self.penalty
        hyperparam_dict["params"] = self.params
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
        


class SingleThresholdStatisticalProfiling(StatisticalProfiling, SingleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, **params):
        super().__init__(**params)
        SingleThresholdMethod.__init__(self)
        self.method_name = "SingleThresholdSP"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)

        
class DoubleThresholdStatisticalProfiling(StatisticalProfiling, DoubleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, **params):
        super().__init__(**params)
        DoubleThresholdMethod.__init__(self)
        self.method_name = "DoubleThresholdSP"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)

        
class SingleThresholdIsolationForest(IsolationForest, SingleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, **params):
        super().__init__(**params)
        SingleThresholdMethod.__init__(self)
        self.method_name = "SingleThresholdIF"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)


class SingleThresholdBinarySegmentation(BinarySegmentation, SingleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, **params):
        super().__init__(**params)
        SingleThresholdMethod.__init__(self)
        self.method_name = "SingleThresholdBS"
        SaveableModel.__init__(self, base_models_path, preprocessing_hash)

        
class DoubleThresholdBinarySegmentation(BinarySegmentation, DoubleThresholdMethod, SaveableModel):
    
    def __init__(self, base_models_path, preprocessing_hash, **params):
        super().__init__(**params)
        DoubleThresholdMethod.__init__(self)
        self.method_name = "DoubleThresholdBS"
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
    
    def __init__(self, base_models_path, preprocessing_hash, method_classes, method_hyperparameter_dict_list, cutoffs_per_method, score_function=f_beta):

        self.is_ensemble = True
        
        self.method_classes = method_classes
        self.method_hyperparameter_dicts = method_hyperparameter_dict_list
        self.cutoffs_per_method = cutoffs_per_method
        self.score_function = f_beta
        self.preprocessing_hash = preprocessing_hash
        
        self.models = [method(base_models_path, preprocessing_hash, **hyperparameters, used_cutoffs=used_cutoffs) for method, hyperparameters, used_cutoffs in zip(method_classes, method_hyperparameter_dict_list, self.cutoffs_per_method)]
        
        self.method_name = " + ".join([model.method_name for model in self.models])
        #self.method_name = "StackEnsemble"
        
        super().__init__(base_models_path, preprocessing_hash)

        
    def fit_transform_predict(self, X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit=True):
        self._scores = []
        temp_scores = []
        self._predictions = []
        for model in self.models:
            scores, predictions = model.fit_transform_predict(X_dfs, y_dfs, label_filters_for_all_cutoffs, base_scores_path, base_predictions_path, overwrite, fit)
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
    def __init__(self, base_models_path, method_classes, method_hyperparameter_dict_list, all_cutoffs, score_function=f_beta):
        cutoffs_per_method = [all_cutoffs]*len(method_classes)
        
        super().__init__(base_models_path, method_classes, method_hyperparameter_dict_list, cutoffs_per_method, score_function=f_beta)