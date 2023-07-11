from typing import List, Tuple, Union

import os
import pickle
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.preprocessing import RobustScaler

from .io_functions import save_dataframe_list


def get_label_filters_for_all_cutoffs(y_df, length_df, all_cutoffs, remove_missing=False, missing_df=None):
    

    uncertain_filter = y_df["label"] == 5
    
    partial_filter = {}
    #Procedure is non-inclusive on lower cutoff
    for cutoffs in all_cutoffs:
        low_cutoff, high_cutoff = cutoffs
        #Only keep rows (timestamps) where the 
        partial_filter[str(cutoffs)] = np.logical_and(length_df["lengths"] > low_cutoff, length_df["lengths"] <= high_cutoff)
        
        #labels_for_all_cutoffs[str(cutoffs)] = y_df.loc[filter_condition[str(cutoffs)], "label"]
        
        #labels_for_all_cutoffs[str(cutoffs)] = (((event_lengths["lengths"] > low_cutoff) & (event_lengths["lengths"] <= high_cutoff)) | event_lengths["lengths"] == 0)
    
    print("full filter construction")
    full_filters = {}
    for cutoffs in all_cutoffs:
        other_cutoffs = list(set(all_cutoffs).difference(set([cutoffs])))
        
        other_partial_filters = [partial_filter[str(c)] for c in other_cutoffs]
        
        length_filter= np.logical_or.reduce(other_partial_filters)
        
        if remove_missing:
            missing_filter = missing_df["missing"] != 0
            full_filters[str(cutoffs)] = np.logical_or.reduce([uncertain_filter, length_filter, missing_filter])
        else:
            full_filters[str(cutoffs)] = np.logical_or(uncertain_filter, length_filter)
        
        
        print(cutoffs)
        print(np.sum(full_filters[str(cutoffs)]))
        
    return full_filters

def get_event_lengths(y_df):
    
        lengths = np.zeros(len(y_df))
        
        event_started = False
        event_start_index = None
        
        for i in range(len(y_df)):
            
            
            if event_started:
                
                #Event ends
                if y_df["label"][i] == 0:
                    event_end_index = i #not inclusive
                    
                    lengths[event_start_index:event_end_index] = event_end_index-event_start_index
                    event_started = False
                #Event continues
                else:
                    pass
            else:
                #Event starts
                if y_df["label"][i] != 0:
                    event_start_index = i
                    event_started = True
                #Event has not started:
                else:
                    pass
                
        #if event has not ended at end of timeseries:
        if event_started:
            event_end_index = i+1 #not inclusive
            
            lengths[event_start_index:event_end_index] = event_end_index-event_start_index
        return pd.DataFrame({"lengths":lengths})

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Match bottom up with substation measurements with linear regression and apply the sign value to the substation measurements.

    Args:
        df (pd.DataFrame): Dataframe with at least the columns M_TIMESTAMP, S_original, BU_original and Flag.

    Returns:
        pd.DataFrame: DataFrame with the columns M_TIMESTAMP, S_original, BU_original, diff_original, S, BU, diff, and missing.
    """
    #Adjust timestamp that suffer from wrong data due to the clock moving when the time period exists in the data.
    try:
        df.loc[(df['M_TIMESTAMP']>='2020-03-29 03:00:00')
              &(df['M_TIMESTAMP']< '2020-03-29 04:00:00'),'Flag'] = "5"
        df.loc[(df['M_TIMESTAMP']>='2020-03-29 03:00:00')
              &(df['M_TIMESTAMP']< '2020-03-29 04:00:00'),'S_original'] = list(df[df['M_TIMESTAMP']<'2020-03-29 00:03:00']['S_original'])[-1]
    except:
        pass
    
    # Calculate difference and add label column.
    df['diff_original'] = df['S_original']-df['BU_original']
        
    # Flag measurement mistakes BU and SO
    # 0 okay
    # 1 measurement missing
    # 2 bottom up missing
    # Flag measurement as missing when 5 times after each other the same value expect 0
    df['S'] = df['S_original'].copy()
    df['missing'] = 0
    df.loc[df['BU_original'].isnull(),'missing'] = 2
    df.loc[(((df['S']==df['S'].shift(1)) & (df['S']==df['S'].shift(2))&(df['S']==df['S'].shift(3))&(df['S']==df['S'].shift(4)))|
           ((df['S']==df['S'].shift(-1)) & (df['S']==df['S'].shift(1))&(df['S']==df['S'].shift(2))&(df['S']==df['S'].shift(3)))|
           ((df['S']==df['S'].shift(-2)) & (df['S']==df['S'].shift(-1))&(df['S']==df['S'].shift(1))&(df['S']==df['S'].shift(2)))|
           ((df['S']==df['S'].shift(-3)) & (df['S']==df['S'].shift(-2))&(df['S']==df['S'].shift(-1))&(df['S']==df['S'].shift(1)))|
           ((df['S']==df['S'].shift(-4)) & (df['S']==df['S'].shift(-3))&(df['S']==df['S'].shift(-2))&(df['S']==df['S'].shift(-1))))&
           df['S']!=0
           ,'missing'] = 1
    
    # Match bottom up with substation measurements for the middle 80% of the values and apply sign to substation measurements
    arr = df[df['missing']==0]
    arr = arr[(arr['diff_original'] > np.percentile(arr['diff_original'],10)) & (arr['diff_original'] < np.percentile(arr['diff_original'],90))]
    
    a, b = match_bottomup_load(bottomup_load=arr['BU_original'], measurements=arr['S_original'])
    df['BU'] = a*df['BU_original']+b
    if df['S_original'].min()>0:
        df['S'] = np.sign(df['BU'])*df['S']
    df['diff'] = df['S']-df['BU']
    
    
    return df[['M_TIMESTAMP', 
               'S_original', 'BU_original', 'diff_original', 
               'S', 'BU', 'diff', 
               'missing']]


def match_bottomup_load(bottomup_load: Union[pd.Series, np.ndarray], measurements: Union[pd.Series, np.ndarray]) -> Tuple[int]:
    """Match bottom up with substation measurements with linear regression and apply the sign value to the substation measurements.

    Args:
        bottomup_load (Union[pd.Series, np.ndarray]): Contains the bottom up load of a substation.
        measurements (Union[pd.Series, np.ndarray]): Contains the measured load of a substation.

    Returns:
        Tuple[int, int]: Optimized parameter a, used to multiply the bottom up load and optimized parameter b, used to add to the bottom up load.
    """
    def calculate_ab(ab: List[int], bottomup_load: Union[pd.Series, np.ndarray], measurements: Union[pd.Series, np.ndarray]):
        a, b  = ab
        if min(measurements) < 0:
            return np.sum(((a*bottomup_load+b)-measurements)**2)

        return np.sum((abs(a*bottomup_load+b)-measurements)**2)

    #Optimize a and b variables of linear regression.
    ab_initial = [1,0] #initial guess: bottomup_load is correct --> a=1, b=0
    ab = opt.minimize(calculate_ab, x0=ab_initial, args=(bottomup_load, measurements))

    #Use a and b to calculate new bottom up load and adjusted measurements.
    a, b = ab.x
    return a, b

def preprocess_per_batch_and_write(X_dfs, y_dfs, intermediates_folder, which_split, preprocessing_type, preprocessing_overwrite, write_csv_intermediates, file_names, all_cutoffs, remove_missing=False):
    #Set preprocessing settings here:
    preprocessed_pickles_folder = os.path.join(intermediates_folder, "preprocessed_data_pickles", which_split)
    preprocessed_csvs_folder = os.path.join(intermediates_folder, "preprocessed_data_csvs", which_split)

    #TODO: preprocess_data needs rework
    # - The following need to be toggles/settings:
    #   - Whether to filter 'Missing' values when they are identical for N subsequent measurements
    #   - Percentiles for sign correction need to be adjustable
    # - The function needs only return a subset of columns (this will save substantially in memory/loading overhead)

    #TODO: Add functionality to preprocess test/validation based on statistics found in train

    #TODO: Name needs to change based on settings (NYI)
    #preprocessing_type = "basic"
    
    preprocessed_file_name = os.path.join(preprocessed_pickles_folder, preprocessing_type + ".pickle")

    if preprocessing_overwrite or not os.path.exists(preprocessed_file_name):
        print("Preprocessing X data")
        X_dfs_preprocessed = [preprocess_data(df) for df in X_dfs]
        
        os.makedirs(preprocessed_pickles_folder, exist_ok = True)
        with open(preprocessed_file_name, 'wb') as handle:
            pickle.dump(X_dfs_preprocessed, handle)
    else:
        print("Loading preprocessed X data")
        with open(preprocessed_file_name, 'rb') as handle:
            X_dfs_preprocessed = pickle.load(handle)

    if write_csv_intermediates:
        print("Writing CSV intermediates: X train data")
        type_preprocessed_csvs_folder = os.path.join(preprocessed_csvs_folder, preprocessing_type)
        save_dataframe_list(X_dfs_preprocessed, file_names, type_preprocessed_csvs_folder, overwrite = preprocessing_overwrite)

    #Preprocess Y_data AKA get the lengths of each event
    event_lengths_pickles_folder = os.path.join(intermediates_folder, "event_length_pickles", which_split)
    event_lengths_csvs_folder = os.path.join(intermediates_folder, "event_length_csvs", which_split)

    preprocessed_file_name = os.path.join(event_lengths_pickles_folder, str(all_cutoffs) + ".pickle")
    if preprocessing_overwrite or not os.path.exists(preprocessed_file_name):
        print("Preprocessing event lengths")
        event_lengths = [get_event_lengths(df) for df in y_dfs]
        
        os.makedirs(event_lengths_pickles_folder, exist_ok = True)
        with open(preprocessed_file_name, 'wb') as handle:
            pickle.dump(event_lengths, handle)
    else:
        print("Loading preprocessed event lengths")
        with open(preprocessed_file_name, 'rb') as handle:
            event_lengths = pickle.load(handle)

    if write_csv_intermediates:
        print("Writing CSV intermediates: event lengths")
        type_event_lengths_csvs_folder = os.path.join(event_lengths_csvs_folder, preprocessing_type)
        save_dataframe_list(event_lengths, file_names, type_event_lengths_csvs_folder, overwrite = preprocessing_overwrite)


    # Use the event lengths to get conditional label filters per cutoff
    label_filters_per_cutoff_pickles_folder = os.path.join(intermediates_folder, "label_filters_per_cutoff_pickles", which_split)
    label_filters_per_cutoff_csvs_folder = os.path.join(intermediates_folder, "label_filters_per_cutoff_csvs", which_split)

    preprocessed_file_name = os.path.join(label_filters_per_cutoff_pickles_folder, str(all_cutoffs) + ".pickle")
    if preprocessing_overwrite or not os.path.exists(preprocessed_file_name):
        print("Preprocessing labels per cutoff")
        label_filters_for_all_cutoffs = [get_label_filters_for_all_cutoffs(y_df, length_df, all_cutoffs, remove_missing=remove_missing) for y_df, length_df, X_df in zip(y_dfs, event_lengths, X_dfs)]
        
        os.makedirs(label_filters_per_cutoff_pickles_folder, exist_ok = True)
        with open(preprocessed_file_name, 'wb') as handle:
            pickle.dump(label_filters_for_all_cutoffs, handle)
    else:
        print("Loading preprocessed labels per cutoff")
        with open(preprocessed_file_name, 'rb') as handle:
            label_filters_for_all_cutoffs = pickle.load(handle)

    if write_csv_intermediates:
        print("Writing CSV intermediates: label filters per cutoff")
        type_label_filters_per_cutoff_csvs_folder = os.path.join(label_filters_per_cutoff_csvs_folder, preprocessing_type)
        save_dataframe_list(label_filters_for_all_cutoffs, file_names, type_label_filters_per_cutoff_csvs_folder, overwrite = preprocessing_overwrite)
        
    preprocessing_parameters = {} #TODO: implement saving of settings for some methods (not basic)
    
    return X_dfs_preprocessed, label_filters_for_all_cutoffs, event_lengths, preprocessing_parameters