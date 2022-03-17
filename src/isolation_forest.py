from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def threshold_scores(y_scores: np.ndarray, threshold:float):
    (y_scores < threshold).astype(float)

def isolation_forest(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply isolation forest method.

    Args:
        df (pd.DataFrame): A DataFrame with at least the columns M_TIMESTAMP, diff_original.

    Returns:
        pd.DataFrame:  Similar to input with added column PRED_IF.
    """
    np.random.seed(31415)
    
    diff_original_notnull_check = df['diff_original'].notnull()
    diff_original_notnull = df[diff_original_notnull_check]['diff_original']

    model =  IsolationForest()
    data = np.array(diff_original_notnull).reshape(-1,1)
    model.fit(data)
    isolation_forest_output = model.decision_function(data)
    isolation_forest_predictions = [1 if x < threshold else 0 for x in isolation_forest_output]

    df['PRED_IF'] = 0
    df.loc[diff_original_notnull_check,'PRED_IF'] = isolation_forest_predictions
    return df
    