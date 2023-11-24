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


def plot_bkps(signal, bkps, **kwargs):
    """
    Adapted from rpt.display for our purposes
    (https://dev.ipol.im/~truong/ruptures-docs/build/html/_modules/ruptures/show/display.html)

    """
    plt.plot(signal, **kwargs)
    
    # color each regime according to breakpoints
    bkps = sorted(bkps)
    alpha = 0.2  # transparency of the colored background

    prev_bkp = 0
    col = "#4286f4"
    
    for bkp in bkps:
        plt.axvspan(max(0, prev_bkp - 0.5), bkp - 0.5, facecolor=col, alpha=alpha)
        prev_bkp = bkp
        
        # cycle through colours
        if col == "#4286f4":
            col = "#f44174"
        else:
            col = "#4286f4"
            
def plot_IFscores(scores, **kwargs):
    plt.plot(scores, **kwargs)
    

def plot_SP(X_df, preds, threshold, file, model_string):
    """
    Plot the predictions and original plot for the statistical profiling method,
    overlay with thresholds and colour appropriately

    Parameters
    ----------
    X_df : dataframe
        dataframe to be plottedd
    preds : dataframe
        dataframe containing the predictions (0 or 1)
    threshold : int or tuple
        threshold that decides which values are classified as outliers
    file : string
        filename of the dataframe
    model_string : string
        current model

    """
    
    # check if doublethresholding was used
    if type(threshold) is tuple:
        lower_threshold, upper_threshold = threshold
    else: 
        lower_threshold, upper_threshold = -threshold, threshold
     
    fig = plt.figure(figsize=(30,16))  
    plt.title("SP, " + model_string + "\n Predictions station: " + file, fontsize=60)
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
    
    plt.axhline(y=lower_threshold, color='black', linestyle='dashed', label = "threshold")
    plt.axhline(y=upper_threshold, color='black', linestyle='dashed')
    
    plt.legend(fontsize=20, loc="lower left")
    
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
    
def plot_BS(X_df, preds, threshold, file, bkps, model_string):
    """
    Plot the predictions and original plot for the binary segmentation method,
    overlay with thresholds

    Parameters
    ----------
    X_df : dataframe
        dataframe to be plottedd
    preds : dataframe
        dataframe containing the predictions (0 or 1)
    threshold : int or tuple
        threshold that decides which values are classified as outliers
    file : string
        filename of the dataframe
    model_string : string
        current model

    """
    
    # check if doublethresholding was used
    if type(threshold) is tuple:
        lower_threshold, upper_threshold = threshold
    else: 
        lower_threshold, upper_threshold = -threshold, threshold
     
    fig = plt.figure(figsize=(30,16))  
    plt.title("BS, " + model_string + "\n Predictions station: " + file, fontsize=60)
    gs = GridSpec(5, 1, figure=fig)
    
    #Diff plot:
    signal = X_df['diff'].values.reshape(-1,1)
    
    ax1 = fig.add_subplot(gs[:4,:])
    plot_bkps(signal, bkps, label="diff")
    sns.set_theme()

    plt.yticks(fontsize=20)
    plt.ylabel("S diff", fontsize=25)
    
    # plot total mean and thresholds
    total_mean = np.mean(signal) # only works for reference point = mean
    plt.axhline(y=total_mean, color='orange', linestyle='-', linewidth=5, label = "Reference Point")
    plt.axhline(y=total_mean + lower_threshold, color='black', linestyle='dashed', label = "threshold")
    plt.axhline(y=total_mean + upper_threshold, color='black', linestyle='dashed')
    
    prev_bkp = 0
    
    # plot the means of each segment
    for bkp in bkps:
        segment = X_df['diff'][prev_bkp:bkp] # define a segment between two breakpoints
        segment_mean = np.mean(segment)
        
        plt.axhline(y=segment_mean, xmin=prev_bkp / len(X_df['diff']), xmax=bkp/len(X_df['diff']), color='r', linestyle='-', linewidth=5, label = 'Mean over segment')
        
        prev_bkp = bkp
    
    # stop repeating labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=20, loc="lower right")
    
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


def plot_IF(X_df, preds, threshold, y_scores, file, model_string):
    """
    Plot the predictions and original plot for the binary segmentation method,
    overlay with thresholds

    Parameters
    ----------
    X_df : dataframe
        dataframe to be plottedd
    preds : dataframe
        dataframe containing the predictions (0 or 1)
    threshold : int
        threshold on the scores that decides which values are classified as outliers
    file : string
        filename of the dataframe
    model_string : string
        current model

    """
     
    fig = plt.figure(figsize=(30,16))  
    plt.title("IF, " + model_string + "\n Predictions station: " + file, fontsize=60)
    gs = GridSpec(5, 1, figure=fig)
    
    # Diff plot:    
    ax1 = fig.add_subplot(gs[:2,:])
    plot_diff(X_df, label = "diff")
    sns.set_theme()

    plt.yticks(fontsize=20)
    plt.ylabel("S diff", fontsize=25)
    
    #plt.legend(fontsize=20, loc="lower left")
    
    # Scores plot:    
    ax2 = fig.add_subplot(gs[2:4,:], sharex=ax1)
    plot_IFscores(y_scores)
    sns.set_theme()
    
    # plot threshold on scores
    plt.axhline(y=threshold, color='black', linestyle='dashed', label = "threshold")
    
    ax2.set_ylabel("Scores", fontsize=25)
    
    plt.legend(fontsize=20, loc="lower left")
    
    # Predictions plot
    ax3 = fig.add_subplot(gs[4,:],sharex=ax1)
    plot_labels(preds, label="label")
    sns.set_theme()
    
    ax3.set_ylabel("Predictions", fontsize=25)
    
    ticks = np.linspace(0,len(X_df["S"])-1, 10, dtype=int)
    plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_df)))
    plt.xlabel("Date", fontsize=25)
        
    #ax2.get_xaxis().set_visible(False)
    #ax2.get_yaxis().set_visible(False)
    
    fig.tight_layout()
    plt.show()

    
def plot_predictions(X_dfs, predictions, dfs_files, model, which_stations = None):
    # select random stations if no stations selected
    if which_stations == None:
        which_stations = np.random.randint(0, len(X_dfs), 3)
    
    for station in which_stations:
        X_df = X_dfs[station]
        preds = predictions[station]
        file = dfs_files[station]
        
        match model.method_name:
            
            case "SingleThresholdSP" :
                threshold = model.scaled_optimal_threshold
                plot_SP(X_df, preds, threshold, file, str(model.get_model_string()))
            
            case "DoubleThresholdSP" :
                threshold = model.scaled_optimal_threshold
                plot_SP(X_df, preds, threshold, file, str(model.get_model_string()))
        
            case "SingleThresholdBS" :
                threshold = model.scaled_optimal_threshold
                bkps = model.breakpoints_list[station]
                plot_BS(X_df, preds, threshold, file, bkps, str(model.get_model_string()))
            
            case "DoubleThresholdBS" :
                threshold = model.scaled_optimal_threshold
                bkps = model.breakpoints_list[station]
                plot_BS(X_df, preds, threshold, file, bkps, str(model.get_model_string()))
            
            case "SingleThresholdIF" :
                threshold = model.optimal_threshold
                y_scores = model.y_scores[station]
                plot_IF(X_df, preds, threshold, y_scores, file, str(model.get_model_string()))