import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection

import ruptures as rpt
import seaborn as sns

import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_BU_original(df, **kwargs):
    plt.plot(df["BU_original"], **kwargs)
    
def plot_S_original(df, **kwargs):
    plt.plot(df["S_original"], **kwargs)
    
def plot_S(df, **kwargs):
    plt.plot(df["S"], **kwargs)
    
def plot_BU(df, **kwargs):
    plt.plot(df["BU"], **kwargs)
    
def plot_diff(df, **kwargs):
    plt.plot(df["diff"], **kwargs)
    
def plot_missing(df, **kwargs):
    cmap = ListedColormap(['white', 'red'])
    plt.imshow(df["missing"].values.reshape(1, len(df["missing"])), cmap=cmap, aspect="auto")
    
def plot_labels(df, **kwargs):
    #cmap = ListedColormap(['white', 'red'])
    #plt.imshow((df["label"]==1).values.reshape(1, len(df["label"])), cmap=cmap, aspect="auto")
    #cmap = ListedColormap(['none', 'green'])
    #plt.imshow((df["label"]==5).values.reshape(1, len(df["label"])), cmap=cmap, aspect="auto")
    plt.plot(df["label"], **kwargs)


def plot_SP(X_df, preds, threshold, file, hyperparameter_string):
    """
    Plot the predictions and original plot for the statistical profiling method,
    overlay with thresholds and colour appropriately

    Parameters
    ----------
    X_df : dataframe
        dataframe to be plottedd
    preds : dataframe
        dataframe containing the predictions (0 or 1)
    optimal_threshold : int or tuple
        threshold that decides which values are classified as outliers
    file : string
        filename of the dataframe
    hyperparameter_string : string
        current hyperparemeters

    """
    
    # check if doublethresholding was used
    if type(threshold) is tuple:
        lower_threshold, upper_threshold = threshold
    else: 
        lower_threshold, upper_threshold = -threshold, threshold
     
    fig = plt.figure(figsize=(30,16))  
    plt.title("SP, " + hyperparameter_string + ", Predictions station: " + file, fontsize=60)
    gs = GridSpec(5, 1, figure=fig)
    
    #Diff plot:       
    ax1 = fig.add_subplot(gs[:4,:])
    
    # preparation for colourmap
    y_colormap = np.linspace(0, len(X_df['diff']) - 1, len(X_df['diff']))
    points = np.array([y_colormap, X_df['diff']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # create a colourmap to paint all points above/below the threshold red
    cmap = ListedColormap(['r', 'b', 'r'])
    norm = BoundaryNorm([-np.inf, lower_threshold, upper_threshold, np.inf], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(X_df["diff"])
    lc.set_linewidth(2)
    
    ax1.add_collection(lc)
    sns.set_theme()
    
    plt.yticks(fontsize=20)
    plt.ylabel("S diff", fontsize=25)
    
    plt.axhline(y=lower_threshold, color='r', linestyle='-')
    plt.axhline(y=upper_threshold, color='r', linestyle='-')
    
    #plt.legend(fontsize=20, loc="lower left")
    
    # Predictions plot
    ax2 = fig.add_subplot(gs[4,:],sharex=ax1)
    plot_labels(preds, label="label")
    sns.set_theme()
    
    ax2.set_ylabel("Predictions", fontsize=25)
    
    ticks = np.linspace(0,len(X_df["S"])-1, 10, dtype=int)
    plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_df)))
    plt.xlabel("Date", fontsize=25)
    
    #ax2.get_xaxis().set_visible(False)
    #ax2.get_yaxis().set_visible(False)
    
    fig.tight_layout()
    plt.show()
    
def plot_BS(X_df, preds, threshold, file, bkps, hyperparameter_string):
    """
    Plot the predictions and original plot for the binary segmentation method,
    overlay with thresholds

    Parameters
    ----------
    X_df : dataframe
        dataframe to be plottedd
    preds : dataframe
        dataframe containing the predictions (0 or 1)
    optimal_threshold : int or tuple
        threshold that decides which values are classified as outliers
    file : string
        filename of the dataframe
    hyperparameter_string : string
        current hyperparemeters

    """
    
    # check if doublethresholding was used
    if type(threshold) is tuple:
        lower_threshold, upper_threshold = threshold
    else: 
        lower_threshold, upper_threshold = -threshold, threshold
     
    fig = plt.figure(figsize=(30,16))  
    plt.title("SP, " + hyperparameter_string + ", Predictions station: " + file, fontsize=60)
    gs = GridSpec(5, 1, figure=fig)
    
    #Diff plot:       
    ax1 = fig.add_subplot(gs[:4,:])
    sns.set_theme()
    
    signal = X_df['diff'].values.reshape(-1,1)
    rpt.display(signal, bkps)
    
    plt.yticks(fontsize=20)
    plt.ylabel("S diff", fontsize=25)
    
    plt.axhline(y=lower_threshold, color='r', linestyle='-')
    plt.axhline(y=upper_threshold, color='r', linestyle='-')
    
    #plt.legend(fontsize=20, loc="lower left")
    
    # Predictions plot
    ax2 = fig.add_subplot(gs[4,:],sharex=ax1)
    plot_labels(preds, label="label")
    sns.set_theme()
    
    ax2.set_ylabel("Predictions", fontsize=25)
    
    ticks = np.linspace(0,len(X_df["S"])-1, 10, dtype=int)
    plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_df)))
    plt.xlabel("Date", fontsize=25)
    
    #ax2.get_xaxis().set_visible(False)
    #ax2.get_yaxis().set_visible(False)
    
    fig.tight_layout()
    plt.show()
    
def plot_predictions(X_dfs, predictions, threshold, dfs_files, current_method, model, hyperparameter_string, which_stations = None):
    # select random stations if no stations selected
    if which_stations == None:
        which_stations = np.random.randint(0, len(X_dfs), 5)
    
    for station in which_stations:
        X_df = X_dfs[station]
        preds = predictions[station]
        file = dfs_files[station]
        
        if current_method == "SingleThresholdSP" or current_method == "DoubleThresholdSP":
            plot_SP(X_df, preds, threshold, file, hyperparameter_string)
        if current_method == "SingleThresholdBS" or current_method == "DoubleThresholdBS":
            bkps = model.breakpoints[station]
            plot_BS(X_df, preds, threshold, file, bkps, hyperparameter_string)