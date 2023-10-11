#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:09:06 2023

@author: rbouman
"""

from src.methods import NaiveStackEnsemble


all_cutoffs = [(0, 24), (24, 288), (288, 4032), (4032, np.inf)]

method_hyperparameter_dict_list = [{'beta':0.12, 'model':'l1','min_size':100, 'jump':10, 'quantiles':(5,95), 'scaling':True, 'penalty':'fused_lasso'},{'quantiles': (5, 95)}]

naive_ensemble = NaiveStackEnsemble([SingleThresholdBinarySegmentation, SingleThresholdStatisticalProfiling], method_hyperparameter_dict_list, all_cutoffs=all_cutoffs)

y_train_predictions_dfs = naive_ensemble.fit_transform_predict(X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, fit=True)


metric = cutoff_averaged_f_beta(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)

minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_train_dfs_preprocessed, y_train_dfs_preprocessed, y_train_predictions_dfs, load_column="S_original")
absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
PRFAUC_table = calculate_PRFAUC_table(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)

print(PRFAUC_table)

#%% test multiple cutoff ensemble


from src.methods import StackEnsemble


cutoffs_per_method = [[(288, 4032), (4032, np.inf)], [(0, 24), (24, 288)]]

method_hyperparameter_dict_list = [{'beta':0.12, 'model':'l1','min_size':100, 'jump':10, 'quantiles':(5,95), 'scaling':True, 'penalty':'fused_lasso'},{'quantiles': (5, 95)}]

ensemble = NaiveStackEnsemble([SingleThresholdBinarySegmentation, SingleThresholdStatisticalProfiling], method_hyperparameter_dict_list, cutoffs_per_method)

y_train_predictions_dfs = ensemble.fit_transform_predict(X_train_dfs_preprocessed, y_train_dfs_preprocessed, label_filters_for_all_cutoffs_train, fit=True)


metric = cutoff_averaged_f_beta(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)

minmax_stats = calculate_unsigned_absolute_and_relative_stats(X_train_dfs_preprocessed, y_train_dfs_preprocessed, y_train_predictions_dfs, load_column="S_original")
absolute_min_differences, absolute_max_differences, relative_min_differences, relative_max_differences = minmax_stats
PRFAUC_table = calculate_PRFAUC_table(y_train_dfs_preprocessed, y_train_predictions_dfs, label_filters_for_all_cutoffs_train, beta)

print(PRFAUC_table)