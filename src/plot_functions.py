import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

import seaborn as sns

import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import RobustScaler

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


def plot_bkps(signal, preds, bkps, ax, **kwargs):
    """
    Adapted from rpt.display for our purposes
    (https://dev.ipol.im/~truong/ruptures-docs/build/html/_modules/ruptures/show/display.html)

    """
    ax.plot(signal, **kwargs)
    
    # color each regime according to breakpoints
    bkps = sorted(bkps)
    alpha = 0.2  # transparency of the colored background

    prev_bkp = 0
    
    for bkp in bkps:
        # select colour to fill section with (red for outlier, blue for normal)
        if preds["label"][bkp - 1] == 1:
            col = "#f44174"
            label = "Predicted as outlier"
        else:
            col = "#4286f4"
            label = None
        
        # colour section
        ax.axvspan(max(0, prev_bkp - 0.5), bkp - 0.5, facecolor=col, alpha=alpha, label = label)
        prev_bkp = bkp
        
        # print breakpoint line
        if bkp != 0 and bkp < len(signal):
            ax.axvline(x=bkp - 0.5, color="k", linewidth= 3, linestyle= "--", label = "Breakpoint")
 
def plot_threshold_colour(x, ax, upper_threshold, lower_theshold = None, colour1 = 'b', colour2 = 'r'):
    """
    Plot a line where the the points outside the thresholds are coloured differently

    Parameters
    ----------
    x : array or series
        the values to be plotted
    lower_theshold : float
        threshold below which the plot is coloured differently
    upper_threshold : float
        threshold below which the plot is coloured differently
    ax : Axs object
        the ax to plot on
    colour1 : string, optional
        the colour of the normal line. The default is 'b'.
    colour2 : string, optional
        the colour of the line outside the thresholds. The default is 'r'.

    """
    # preparation for colourmap
    y_colormap = np.linspace(0, len(x) - 1, len(x))
    points = np.array([y_colormap, x]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # create a colourmap to paint all points above/below the threshold red
    if lower_theshold != None:
        cmap = ListedColormap([colour2, colour1, colour2])
        norm = BoundaryNorm([-np.inf, lower_theshold, upper_threshold, np.inf], cmap.N)
    else:
        cmap = ListedColormap([colour1, colour2])
        norm = BoundaryNorm([-np.inf, upper_threshold, np.inf], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x)
    lc.set_linewidth(2)
    
    ax.add_collection(lc)
    

def plot_SP(X_df, preds, threshold, file, model_string, pretty_plot):
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
    pretty_plot : boolean
        indicates whether to print the plot without title and predictions

    """
    
    # check if doublethresholding was used
    if type(threshold) is tuple:
        lower_threshold, upper_threshold = threshold
    else: 
        lower_threshold, upper_threshold = -threshold, threshold
     
    fig = plt.figure(figsize=(30,16))  
    
    
    if pretty_plot:
        gs = GridSpec(4, 1, figure=fig)
    else:
        plt.title("SPC, " + model_string + "\n Predictions station: " + file, fontsize=60)
        gs = GridSpec(5, 1, figure=fig)
    
    # diff plot coloured correctly:       
    ax1 = fig.add_subplot(gs[:4,:])
    plot_threshold_colour(X_df["diff"], ax1, upper_threshold=upper_threshold, lower_theshold=lower_threshold)
    sns.set_theme()
    
    plt.yticks(fontsize=20)
    plt.ylabel("Scaled difference factor", fontsize=25)
    
    # plot thresholds
    threshold_handle = plt.axhline(y=lower_threshold, color='black', linestyle='dashed', label = "threshold")
    plt.axhline(y=upper_threshold, color='black', linestyle='dashed')
    
    # helper to add red colour to legend
    red_handle = mpatches.Patch(color='red', label='Predicted as outlier')
    
    plt.legend(handles=[threshold_handle, red_handle], fontsize=20, loc="lower left")
    
    # Predictions plot
    if not pretty_plot:
        ax2 = fig.add_subplot(gs[4,:],sharex=ax1)
        plot_labels(preds, label="label")
        sns.set_theme()
        
        ax2.set_ylabel("Predictions", fontsize=25)
    
    ticks = np.linspace(0,len(X_df["S"])-1, 10, dtype=int)
    plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_df)))
    plt.xlabel("Date", fontsize=25)
    
    fig.tight_layout()
    plt.show()
    
def plot_BS(X_df, preds, threshold, file, model, model_string, pretty_plot):
    """
    Plot the predictions and original plot for the binary segmentation method,
    overlay with thresholds

    Parameters
    ----------
    X_df : dataframe
        dataframe to be plotted, contains column named "diff"
    preds : dataframe
        dataframe containing the predictions (0 or 1)
    threshold : int or tuple
        threshold that decides which values are classified as outliers
    file : string
        filename of the dataframe
    model_string : string
        current model
    pretty_plot : boolean
        indicates whether to print the plot without title and predictions

    """
    
    # check if doublethresholding was used
    if type(threshold) is tuple:
        lower_threshold, upper_threshold = threshold
    else: 
        lower_threshold, upper_threshold = -threshold, threshold
     
    fig = plt.figure(figsize=(30,16))  
    
    if pretty_plot:
        gs = GridSpec(4, 1, figure=fig)
    else:
        plt.title("BS, " + model_string + "\n Predictions station: " + file, fontsize=60)
        gs = GridSpec(5, 1, figure=fig)
    
    #Diff plot:    
    bkps = model.get_breakpoints(X_df["diff"].values.reshape(-1,1))
    
    ax1 = fig.add_subplot(gs[:4,:])
    plot_bkps(X_df['diff'], preds, bkps, ax1)
    sns.set_theme()

    plt.yticks(fontsize=20)
    if model.scaling:
        plt.ylabel("Scaled difference factor", fontsize=25)
    else:
        plt.ylabel("Difference factor", fontsize=25)
    
    # plot total mean and thresholds
    if model.reference_point == "mean": # only works for reference point = mean
        total_mean = np.mean(X_df['diff']) 
        plt.axhline(y=total_mean, color='orange', linestyle='-', linewidth=2, label = "Reference Point")
        plt.axhline(y=total_mean + lower_threshold, color='black', linestyle='dashed', label = "threshold")
        plt.axhline(y=total_mean + upper_threshold, color='black', linestyle='dashed')
    else:
        raise Exception("Only use this function when plotting on models that use the mean as reference-point") 
        
    prev_bkp = 0
    
    # plot the means of each segment
    for bkp in bkps:
        segment = X_df['diff'][prev_bkp:bkp] # define a segment between two breakpoints
        segment_mean = np.mean(segment)
        
        plt.axhline(y=segment_mean, xmin=prev_bkp / len(X_df['diff']), xmax=bkp/len(X_df['diff']), color='r', linestyle='-', linewidth=2, label = 'Mean over segment')
        
        prev_bkp = bkp
    
    # stop repeating labels for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=20, loc="lower right")
    
    # Predictions plot
    if not pretty_plot:
        ax2 = fig.add_subplot(gs[4,:],sharex=ax1)
        plot_labels(preds, label="label")
        sns.set_theme()
        
        ax2.set_ylabel("Predictions", fontsize=25)
    
    ticks = np.linspace(0,len(X_df["S"])-1, 10, dtype=int)
    plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_df)))
    plt.xlabel("Date", fontsize=25)
    
    fig.tight_layout()
    plt.show()


def plot_IF(X_df, preds, threshold, file, model, model_string, pretty_plot):
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
    pretty_plot : boolean
        indicates whether to print the plot without title and predictions

    """
     
    fig = plt.figure(figsize=(30,16))
    
    if pretty_plot:
        gs = GridSpec(4, 1, figure=fig)
    else:
        plt.title("IF, " + model_string + "\n Predictions station: " + file, fontsize=60)
        gs = GridSpec(5, 1, figure=fig)
    
    # Diff plot:    
    ax1 = fig.add_subplot(gs[:2,:])
    plot_diff(X_df)
    sns.set_theme()

    plt.yticks(fontsize=20)
    plt.ylabel("Difference factor", fontsize=25)    
    
    # scores plot, colouring the predicted outliers red   
    ax2 = fig.add_subplot(gs[2:4,:], sharex=ax1)
    # calculate y_scores
    y_scores = model.get_IF_scores(X_df)
    plot_threshold_colour(y_scores, ax2, threshold)
    sns.set_theme()
    
    # plot threshold on scores
    threshold_handle = plt.axhline(y=threshold, color='black', linestyle='dashed', label = "threshold")
    
    ax2.set_ylabel("Scores", fontsize=25)
    
    # helper to add red colour to legend
    red_handle = mpatches.Patch(color='red', label='Predicted as outlier')
    plt.legend(handles=[threshold_handle, red_handle], fontsize=20, loc="lower left")
    
    # Predictions plot
    if not pretty_plot:
        ax3 = fig.add_subplot(gs[4,:],sharex=ax1)
        plot_labels(preds, label="label")
        sns.set_theme()
        
        ax3.set_ylabel("Predictions", fontsize=25)
    
    ticks = np.linspace(0,len(X_df["S"])-1, 10, dtype=int)
    plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_df)))
    plt.xlabel("Date", fontsize=25)
    
    fig.tight_layout()
    plt.show()
    
def plot_predictions(X_dfs, predictions, dfs_files, model, pretty_plot = False, which_stations = None, n_stations = 3):
    """
    Plot the predictions made by a specific model in a way that makes sense for the method

    Parameters
    ----------
    X_dfs : list of dataframes
        the dataframes on which the predictions were made
    predictions : list of dataframes
        the dataframes with the predictions (0 and 1s), must be in the same order as X_dfs
    dfs_files : list of strings
        the file names of the dataframes, must be in the same order as X_dfs
    model : a SaveableModel object
        the model used to predict on the data
    pretty_plot : boolean, optional
        decides whether to create plots without excess information. The default is False.
    which_stations : list of ints, optional
        the indices of the dataframes to be plotted. If None, select 3 random stations

    """
    # select random stations if no stations selected
    if which_stations == None:
        which_stations = np.random.randint(0, len(X_dfs), n_stations)
    
    for station in which_stations:
        X_df = X_dfs[station]
        preds = predictions[station]
        file = dfs_files[station]
        
        # find model used
        match model.method_name:
            
            case "SingleThresholdSPC" :
                threshold = model.optimal_threshold
                scaled_df = scale_diff_data(X_df, model.quantiles)
                plot_SP(scaled_df, preds, threshold, file, str(model.get_model_string()), pretty_plot)
            
            case "DoubleThresholdSPC" :
                threshold = model.optimal_threshold
                scaled_df = scale_diff_data(X_df, model.quantiles)
                plot_SP(scaled_df, preds, threshold, file, str(model.get_model_string()), pretty_plot)
        
            case "SingleThresholdBS" :
                threshold = model.optimal_threshold
                if model.scaling:
                    X_df = scale_diff_data(X_df, model.quantiles)
                plot_BS(X_df, preds, threshold, file, model, str(model.get_model_string()), pretty_plot)
            
            case "DoubleThresholdBS" :
                threshold = model.optimal_threshold
                if model.scaling:
                    X_df = scale_diff_data(X_df, model.quantiles)
                plot_BS(X_df, preds, threshold, file, model, str(model.get_model_string()), pretty_plot)
            
            case "SingleThresholdIF" :
                threshold = model.optimal_threshold
                if model.scaling:
                    X_df = scale_diff_data(X_df, model.quantiles)
                plot_IF(X_df, preds, threshold, file, model, str(model.get_model_string()), pretty_plot)
                
def scale_diff_data(df, quantiles):
    """
    Scale the "diff" column of a dataframe and return a changed copy

    Parameters
    ----------
    df : dataframe
        dataframe to be scaled
    quantiles : tuple of ints
        the quantiles used to scale the data

    Returns
    -------
    df_copy : dataframe
        a copy of the original dataframe with the "diff" column scaled

    """
    scaler = RobustScaler(quantile_range=quantiles)
    scaled_diff = pd.DataFrame(scaler.fit_transform(df["diff"].values.reshape(-1,1)))
    
    # make copy to not change original df
    df_copy = df.copy()
    
    df_copy["diff"] = scaled_diff
    return df_copy
    