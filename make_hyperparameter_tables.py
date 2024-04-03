#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:57:04 2024

@author: rbouman
"""

import pandas as pd
import os

dataset = "OS_data" #alternatively: route_data
# data_folder = os.path.join("data", dataset)
# result_folder = os.path.join("results", dataset)
# intermediates_folder = os.path.join("intermediates", dataset)
# model_folder = os.path.join("saved_models", dataset)

table_folder = os.path.join("Tables", dataset)



beta = 1.5
#For IF, pass sequence of dicts to avoid useless hyperparam combos (such as scaling=True, forest_per_station=True)
IF_per_station_hyperparameters = {"n_estimators": [1000], 
                                                 "forest_per_station":[True], 
                                                 "scaling":[False], 
                                                 "score_function_kwargs":[{"beta":beta}],
                                                 "threshold_strategy":["Symmetrical"]}

IF_over_all_hyperparameters = {"n_estimators": [1000], 
                                              "forest_per_station":[False], 
                                              "scaling":[True], 
                                              "quantiles":[(10,90), (15, 85), (20,80)], 
                                              "score_function_kwargs":[{"beta":beta}],
                                              "threshold_strategy":["Symmetrical"]}


SPC_hyperparameters = {"quantiles":[(10,90), (15, 85), (20,80)], 
                                      "score_function_kwargs":[{"beta":beta}],
                                      "threshold_strategy":["Symmetrical", "Asymmetrical"]}


BS_hyperparameters = {"beta": [0.005, 0.008, 0.015, 0.05, 0.08, 0.12], 
                                     "model": ['l1'], 
                                     'min_size': [150, 200, 288], 
                                     "jump": [5, 10], 
                                     "quantiles": [(10,90), (15, 85), (20,80)], 
                                     "scaling": [True], 
                                     "penalty": ['L1'], 
                                     "reference_point":["mean", "median", "longest\_median", "longest\_mean"],
                                     "score_function_kwargs":[{"beta":beta}],
                                     "threshold_strategy":["Symmetrical", "Asymmetrical"]}

#dict to go from internal name to name in paper: (if None, that means it should not be printed)
hyperparameter_translation_dict = {"n_estimators":"$n_{\\textrm{estimators}}$",
                                   "quantiles":"($q_{\\textrm{lower}}\\%$, $q_{\\textrm{upper}}\\%$)",
                                   "beta":"$\\beta$",
                                   "model":None,
                                   "min_size":"$l$",
                                   "jump":"$j$",
                                   "reference_point":"$\\textit{reference\\_point}$",
                                   "forest_per_station":None,
                                   "scaling":None,
                                   "score_function_kwargs":None,
                                   "penalty":"$C$",
                                   "threshold_strategy":"Threshold strategy"}

hyperparams_per_method = {"IF per station":IF_per_station_hyperparameters, "IF over all stations":IF_over_all_hyperparameters, "SPC":SPC_hyperparameters, "BS":BS_hyperparameters}

column_names = ["Method", "Hyperparameter", "Hyperparameter values"]
hyperparameter_list = []
for method_name in hyperparams_per_method:

    for hyperparameter in hyperparams_per_method[method_name]:

        hyperparameter_name = hyperparameter_translation_dict[hyperparameter]
        if hyperparameter_name is not None:
            row = {column_names[0]:method_name, column_names[1]:hyperparameter_name, column_names[2]:hyperparams_per_method[method_name][hyperparameter]}

            hyperparameter_list.append(row)
            
hyperparameter_df = pd.DataFrame(hyperparameter_list, columns=column_names)

hyperparameter_df_index = pd.MultiIndex.from_frame(hyperparameter_df[["Method", "Hyperparameter"]])

hyperparameter_df = pd.DataFrame(hyperparameter_df["Hyperparameter values"])
hyperparameter_df.index = hyperparameter_df_index

hyperparameter_df.to_latex(buf=os.path.join(table_folder, "evaluated_hyperparameters.tex"), escape=False, multirow=True)