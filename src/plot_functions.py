import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
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
    
def plot_predictions(X_dfs, predictions, dfs_files, current_method, current_hyperparameters, which_stations = None):
    # select random stations if no stations selected
    if which_stations == None:
        which_stations = np.random.randint(0, len(X_dfs), 5)
    
    for station in which_stations:  
        X_df = X_dfs[station]
        preds = predictions[station]
        
        fig = plt.figure(figsize=(30,16)) # add DPI=300+ in case some missing points don't show up'    
        plt.title(current_method + ", " + current_hyperparameters + ", Predictions station: " + dfs_files[station], fontsize=60)
        gs = GridSpec(10, 1, figure=fig)
        
        #S/BU original plot:
        ax1 = fig.add_subplot(gs[0:3,:])
        sns.set_theme()
        plot_S_original(X_df, label="S original")
        plot_BU_original(X_df, label="BU original")
        
        plt.legend(fontsize=20)
        
        plt.yticks(fontsize=20)
        plt.ylabel("S", fontsize=25)
        
        ax1.get_xaxis().set_visible(False)
        
        #S/BU plot:
        ax2 = fig.add_subplot(gs[3:6,:], sharex=ax1)
        sns.set_theme()
        plot_S(X_df, label="S")
        plot_BU(X_df, label="BU")
        
        plt.legend(fontsize=20)
        
        plt.yticks(fontsize=20)
        plt.ylabel("S", fontsize=25)
        
        ax2.get_xaxis().set_visible(False)
        
        #Label plot:
        ax3 = fig.add_subplot(gs[6,:],sharex=ax1)
        plot_labels(preds, label="label")
        sns.set_theme()
        
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        
        #plt.colorbar()
        
        #Diff plot:
        ax4 = fig.add_subplot(gs[7:,:], sharex=ax1)
        plot_diff(X_df, label="S-BU")
        sns.set_theme()
        
        plt.yticks(fontsize=20)
        plt.ylabel("S diff", fontsize=25)
        
        ticks = np.linspace(0,len(X_df["S"])-1, 10, dtype=int)
        plt.xticks(ticks=ticks, labels=X_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=20)
        plt.xlim((0, len(X_df)))
        plt.xlabel("Date", fontsize=25)
        
        plt.legend(fontsize=20, loc="lower left")
        
        fig.tight_layout()
        plt.show()