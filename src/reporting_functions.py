#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:18:44 2023

@author: rbouman
"""

def print_metrics_and_stats(metric, PRFAUC_table, keep_NA=False):

    print("metric:" )
    print(metric)
    print("PRF table:")
    print(PRFAUC_table)

def bootstrap_stats_to_printable(mean_table, std_table):
    mean_table_string = mean_table.applymap("{0:.4f}".format)
    std_table_string = std_table.applymap("{0:.4f}".format)
    
    return mean_table_string+"±"+std_table_string
