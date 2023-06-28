import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import seaborn as sns

import pandas as pd
import numpy as np


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
    cmap = ListedColormap(['white', 'red'])
    plt.imshow((df["label"]==1).values.reshape(1, len(df["label"])), cmap=cmap, aspect="auto")
    cmap = ListedColormap(['none', 'green'])
    plt.imshow((df["label"]==5).values.reshape(1, len(df["label"])), cmap=cmap, aspect="auto")