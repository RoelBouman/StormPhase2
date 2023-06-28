from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.preprocessing import RobustScaler

def get_event_lengths(y_df):
    
        lengths = np.zeros(len(y_df))
        
        event_started = False
        event_start_index = None
        
        for i in range(len(y_df)):
            
            
            if event_started:
                
                #Event ends
                if y_df["label"][i] != 1:
                    event_end_index = i #not inclusive
                    
                    lengths[event_start_index:event_end_index] = event_end_index-event_start_index
                    event_started = False
                #Event continues
                else:
                    pass
            else:
                #Event starts
                if y_df["label"][i] == 1:
                    event_start_index = i
                    event_started = True
                #Event has not started:
                else:
                    pass
                
        #if event has not ended at end of timeseries:
        if event_started:
            event_end_index = i+1 #not inclusive
            
            lengths[event_start_index:event_end_index] = event_end_index-event_start_index
        return lengths

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Match bottom up with substation measurements with linear regression and apply the sign value to the substation measurements.

    Args:
        df (pd.DataFrame): Dataframe with at least the columns M_TIMESTAMP, S_original, BU_original and Flag.

    Returns:
        pd.DataFrame: DataFrame with the columns M_TIMESTAMP, S_original, BU_original, diff_original, S, BU, diff, diff_robust and missing.
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
    
    # Robust scaled difference
    df['diff_robust'] = RobustScaler().fit_transform(np.array(df['diff']).reshape(-1, 1))
    
    return df[['M_TIMESTAMP', 
               'S_original', 'BU_original', 'diff_original', 
               'S', 'BU', 'diff', 
               'missing', 'diff_robust']]


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