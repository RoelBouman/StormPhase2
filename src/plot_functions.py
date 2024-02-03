import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

from datetime import datetime, timedelta

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

def plot_TP_FP_FN(y_df, preds, opacity, ax):
    true_values = y_df["label"]
    predictions = preds["label"]
    
    true_values[true_values == 5] = 0 # set all 5s to zero 
    
    TP = np.logical_and(true_values, predictions) # AND between the two arrays
    FP = np.logical_and(np.logical_not(true_values), predictions) # AND with negated true values
    FN = np.logical_and(true_values, np.logical_not(predictions)) # AND with negated predictions
    
    TP_index = np.where(TP == 1)[0]
    FP_index = np.where(FP == 1)[0]
    FN_index = np.where(FN == 1)[0]
    
    
    for i in TP_index:
        ax.axvline(i, color='g', linewidth=3, alpha=opacity)
    for i in FP_index:
        ax.axvline(i, color='y', linewidth=3, alpha=opacity)
    for i in FN_index:
        ax.axvline(i, color='c', linewidth=3, alpha=opacity)
        
    
    TP_handle = mpatches.Patch(color='g', label='True Positive')
    FP_handle = mpatches.Patch(color='y', label='False Positive')
    FN_handle = mpatches.Patch(color='c', label='False Negative')

    
    legend1 = plt.legend(handles=[TP_handle, FP_handle, FN_handle], fontsize=20, loc="upper left")

    ax.add_artist(legend1)

def plot_bkps(signal, y_df, preds, bkps, show_TP_FP_FN, opacity, ax, **kwargs):
    """
    Adapted from rpt.display for our purposes
    (https://dev.ipol.im/~truong/ruptures-docs/build/html/_modules/ruptures/show/display.html)

    """
    # colour background according to TP,FP,FN
    if show_TP_FP_FN:
        plot_TP_FP_FN(y_df, preds, opacity, ax)
    
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
        
        # colour section if not coloured according to TP,FP,FN
        if not show_TP_FP_FN:
            ax.axvspan(max(0, prev_bkp - 0.5), bkp - 0.5, facecolor=col, alpha=alpha, label = label)
        prev_bkp = bkp
        
        # print breakpoint line
        if bkp != 0 and bkp < len(signal):
            ax.axvline(x=bkp - 0.5, color="k", linewidth= 3, linestyle= "--", label = "Breakpoint")
 
def plot_threshold_colour(values, dates, ax, upper_threshold, lower_threshold = None, colour1 = 'b', colour2 = 'r'):
    """
    Plot a line where the the points outside the thresholds are coloured differently

    Parameters
    ----------
    values : array of floats
        the values to be plotted
    dates : array of strings that can be turned into datetime objects
        the timestamps belonging to every value
    ax : Axs object
            the ax to plot on
    upper_threshold : float
        threshold above which the plot is coloured differently
    lower_threshold : float
        threshold below which the plot is coloured differently
    colour1 : string, optional
        the colour of the normal line. The default is 'b'.
    colour2 : string, optional
        the colour of the line outside the thresholds. The default is 'r'.

    Returns
    ---------
    dates : array
        new dates array with dates added that represent the midpoints
        between points that cross the threshold
    """
    # transform strings to datetime objects
    if len(dates[0]) == 19: # if including seconds
        dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in dates]
    elif len(dates[0]) == 16: # if excluding seconds
        dates = [datetime.strptime(date, '%Y-%m-%d %H:%M') for date in dates]
    else:
        raise Exception("Incorrect timestamp formatting")
       
    # adjust the data to make sure colouring is done correctly         
    dates, values = prepare_threshold_data(dates, values, upper_threshold, lower_threshold)    
    
    # preparation for colourmap
    y_colormap = np.linspace(0, len(values) - 1, len(values))
    points = np.array([y_colormap, values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # create a colourmap to paint all points above/below the threshold red
    if lower_threshold != None:
        cmap = ListedColormap([colour2, colour1, colour2])
        norm = BoundaryNorm([-np.inf, lower_threshold, upper_threshold, np.inf], cmap.N)
    else:
        cmap = ListedColormap([colour1, colour2])
        norm = BoundaryNorm([-np.inf, upper_threshold, np.inf], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(values)
    lc.set_linewidth(2)
    
    ax.add_collection(lc)
    
    return dates

def prepare_threshold_data(x, y, upper_threshold, lower_threshold):
    """
    Adjust the data by finding all crossover points and adding them
    to the x and y data

    Parameters
    ----------
    x : array of datetimes
        datetimes corresponding the values in y
    y : array of floats
        the values
    upper_threshold : float
        the upper threshold
    lower_threshold : float or None
        the lower threshold, does not exist when None

    Returns
    -------
    x : array of datetimes
        new array with the dates corresponding to the midpoints added
    y : array of floats
        new array with values of the midpoints added

    """
    
    crossovers = []
    
    # find all two points that cross over the thresholds
    for i, (first,second) in enumerate(zip(y[:-1], y[1:])):
        if first < upper_threshold < second or second < upper_threshold < first:
            # save time of first value, the two values themselves, and the threshold value
            crossovers.append([x[i], first, second, upper_threshold])
        elif lower_threshold != None and (first < lower_threshold < second or second < lower_threshold < first):
            crossovers.append([x[i], first, second, lower_threshold])
       
    # if no crossovers found, return
    if len(crossovers) == 0:
        return x, y
       
    midpoints = [] 
    
    # calculate all the midpoints
    for time, first, second, threshold in crossovers:
        y_new = threshold 
            
        if first < second:
            # slightly decrease the y-value so that it is just above the threshold        
            y_new += 0.00001
            x_new = np.interp(y_new, [first, second], [0, 15])
        else:
            # slightly decrease the y-value so that it is just below the threshold   
            y_new -= 0.00001
            x_new = np.interp(y_new, [second, first], [0, 15])
        
        # create the new timestamp using the interpolated x and adding it to the time            
        new_time = time + timedelta(minutes = int(x_new), seconds = int(60 * (x_new - int(x_new))))
        midpoints.append([new_time, y_new])
    
    midpoints = np.array(midpoints)
            
    x = np.concatenate((x, midpoints[:,0]))
    y = np.concatenate((y, midpoints[:,1]))
        
    # sort both arrays by time
    y = [v for _, v in sorted(zip(x, y))]
    x = sorted(x) 
    
    return x, y
    
    
def plot_SP(X_df, y_df, preds, threshold, file, model_string, show_TP_FP_FN, opacity_TP, pretty_plot):
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
    
    # colour background according to TP,FP,FN
    if show_TP_FP_FN:
        plot_TP_FP_FN(y_df, preds, opacity_TP, ax1)
    
    dates = plot_threshold_colour(np.array(X_df["diff"]), np.array(X_df["M_TIMESTAMP"]), ax1, upper_threshold=upper_threshold, lower_threshold=lower_threshold)
    sns.set_theme()
    
    plt.yticks(fontsize=20)
    plt.ylabel("Scaled difference vector", fontsize=25)
    
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
    
    ticks = np.linspace(0,len(X_df)-1, 10, dtype=int)
    plt.xticks(ticks=ticks, labels=pd.Series(dates).iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_df)))
    plt.xlabel("Date", fontsize=25)
    
    fig.tight_layout()
    plt.show()
    
def plot_BS(X_df, y_df, preds, threshold, file, model, model_string, show_TP_FP_FN, opacity_TP, pretty_plot):
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
    plot_bkps(X_df['diff'], y_df, preds, bkps, show_TP_FP_FN, opacity_TP, ax1)
    sns.set_theme()

    plt.yticks(fontsize=20)
    if model.scaling:
        plt.ylabel("Scaled difference vector", fontsize=25)
    else:
        plt.ylabel("Difference vector", fontsize=25)
    
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


def plot_IF(X_df, y_df, preds, threshold, file, model, model_string, show_TP_FP_FN, opacity_TP, pretty_plot):
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
    
    # colour background according to TP,FP,FN
    if show_TP_FP_FN:
        plot_TP_FP_FN(y_df, preds, opacity_TP, ax1)
    
    plot_diff(X_df)
    sns.set_theme()

    plt.yticks(fontsize=20)
    plt.ylabel("Difference vector", fontsize=25)    
    
    # scores plot, colouring the predicted outliers red   
    ax2 = fig.add_subplot(gs[2:4,:], sharex=ax1)
    # calculate y_scores
    y_scores = model.get_IF_scores(X_df)
    dates = plot_threshold_colour(np.array(y_scores), np.array(X_df["M_TIMESTAMP"]), ax2, upper_threshold=threshold)
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
    
    ticks = np.linspace(0,len(X_df)-1, 10, dtype=int)
    plt.xticks(ticks=ticks, labels=pd.Series(dates).iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_df)))
    plt.xlabel("Date", fontsize=25)
    
    fig.tight_layout()
    plt.show()
    
def plot_predictions(X_dfs, y_dfs, predictions, dfs_files, model, show_TP_FP_FN = True, opacity_TP = 0.3, pretty_plot = False, which_stations = None, n_stations = 3):
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
        y_df = y_dfs[station]
        preds = predictions[station]
        file = dfs_files[station]
        
        # find model used
        match model.method_name:
            
            case "SingleThresholdSPC" :
                threshold = model.optimal_threshold
                scaled_df = scale_diff_data(X_df, model.quantiles)
                plot_SP(scaled_df, y_df, preds, threshold, file, str(model.get_model_string()), show_TP_FP_FN, opacity_TP, pretty_plot)
            
            case "DoubleThresholdSPC" :
                threshold = model.optimal_threshold
                scaled_df = scale_diff_data(X_df, model.quantiles)
                plot_SP(scaled_df, y_df, preds, threshold, file, str(model.get_model_string()), show_TP_FP_FN, opacity_TP, pretty_plot)
        
            case "SingleThresholdBS" :
                threshold = model.optimal_threshold
                if model.scaling:
                    X_df = scale_diff_data(X_df, model.quantiles)
                plot_BS(X_df, y_df, preds, threshold, file, model, str(model.get_model_string()), show_TP_FP_FN, opacity_TP, pretty_plot)
            
            case "DoubleThresholdBS" :
                threshold = model.optimal_threshold
                if model.scaling:
                    X_df = scale_diff_data(X_df, model.quantiles)
                plot_BS(X_df, y_df, preds, threshold, file, model, str(model.get_model_string()), show_TP_FP_FN, opacity_TP, pretty_plot)
            
            case "SingleThresholdIF" :
                threshold = model.optimal_threshold
                if model.scaling:
                    X_df = scale_diff_data(X_df, model.quantiles)
                plot_IF(X_df, y_df, preds, threshold, file, model, str(model.get_model_string()), show_TP_FP_FN, opacity_TP, pretty_plot)
                
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
    