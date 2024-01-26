#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:18:44 2023

@author: rbouman
"""
import numpy as np

def print_metrics_and_stats(metric, minmax_stats, PRFAUC_table, keep_NA=False):
        absolute_min_differences, absolute_max_differences, _, _ = minmax_stats
        
        if keep_NA:
            average_min_difference = np.mean(absolute_min_differences)
            average_max_difference = np.mean(absolute_max_differences)
            
            max_min_difference = np.max(absolute_min_differences)
            max_max_difference = np.max(absolute_max_differences)
        else:
            average_min_difference = np.nanmean(absolute_min_differences)
            average_max_difference = np.nanmean(absolute_max_differences)
            
            max_min_difference = np.nanmax(absolute_min_differences)
            max_max_difference = np.nanmax(absolute_max_differences)
        
        print("metric:" )
        print(metric)
        print("PRF table:")
        print(PRFAUC_table)
        print("Average differences:")
        print("Min:")
        print(average_min_difference)
        print("Max:")
        print(average_max_difference)
        print("Max differences:")
        print("Min:")
        print(max_min_difference)
        print("Max:")
        print(max_max_difference)