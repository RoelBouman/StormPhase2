#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rbouman
"""
import numpy as np

def filter_dfs_to_array(dfs, df_filters):
    #filter defines what to exclude, so, True is excluded
    filtered_dfs = [df[np.logical_not(df_filter)] for df, df_filter in zip(dfs, df_filters)]
    
    return np.concatenate(filtered_dfs)