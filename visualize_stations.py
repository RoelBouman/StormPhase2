#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from src.plot_functions import *



#%% define environment variables
data_folder = "data"
result_folder = "results"
intermediates_folder = "intermediates"

preprocessed_folder = "preprocessed_data_csvs"
preprocessing_type = "basic"

#%% define runtime variables
plot_station_IDs = ["1", "3"]

n_xlabels = 10


#%%
train_IDs = os.listdir(os.path.join(data_folder, "Train", "X"))
test_IDs = os.listdir(os.path.join(data_folder, "Test", "X"))
validation_IDs = os.listdir(os.path.join(data_folder, "Validation", "X"))

all_station_IDs = train_IDs + test_IDs + validation_IDs

train_ID_dict = {ID.replace(".csv", ""): "Train" for ID in train_IDs}
test_ID_dict = {ID.replace(".csv", ""): "Test" for ID in test_IDs}
validation_ID_dict = {ID.replace(".csv", ""): "Validation" for ID in validation_IDs}

#fastest dict merge: https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one
station_dataset_dict = dict(train_ID_dict, **test_ID_dict)
station_dataset_dict.update(validation_ID_dict)

for station_ID in plot_station_IDs:
    print(station_ID)
    print("present in set: " + station_dataset_dict[station_ID])
    
    X_df = pd.read_csv(os.path.join(data_folder, station_dataset_dict[station_ID], "X", station_ID + ".csv"))
    y_df = pd.read_csv(os.path.join(data_folder, station_dataset_dict[station_ID], "y", station_ID + ".csv"))
    
    X_preprocessed_df = pd.read_csv(os.path.join(intermediates_folder, preprocessed_folder, station_dataset_dict[station_ID], preprocessing_type, station_ID + ".csv"))
    
    fig = plt.figure(figsize=(30,16)) # add DPI=300+ in case some missing points don't show up'    
    plt.title("Station: " + station_ID, fontsize=60)
    gs = GridSpec(10, 1, figure=fig)
    
    #S/BU original plot:
    ax1 = fig.add_subplot(gs[0:3,:])
    sns.set_theme()
    plot_S_original(X_preprocessed_df, label="S original")
    plot_BU_original(X_preprocessed_df, label="BU original")
    
    plt.legend(fontsize=20)
    
    plt.yticks(fontsize=20)
    plt.ylabel("S", fontsize=25)
    
    ax1.get_xaxis().set_visible(False)
    
    #S/BU plot:
    ax2 = fig.add_subplot(gs[3:6,:], sharex=ax1)
    sns.set_theme()
    plot_S(X_preprocessed_df, label="S")
    plot_BU(X_preprocessed_df, label="BU")
    
    plt.legend(fontsize=20)
    
    plt.yticks(fontsize=20)
    plt.ylabel("S", fontsize=25)
    
    ax2.get_xaxis().set_visible(False)
    
    #Missing plot:
    ax3 = fig.add_subplot(gs[6,:],sharex=ax1)
    plot_missing(X_preprocessed_df, label="missing", origin="lower")
    sns.set_theme()
    
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    #plt.colorbar()
    
    #Diff plot:
    ax4 = fig.add_subplot(gs[7:,:], sharex=ax1)
    plot_diff(X_preprocessed_df, label="S-BU")
    sns.set_theme()
    
    plt.yticks(fontsize=20)
    plt.ylabel("S diff", fontsize=25)
    
    ticks = np.linspace(0,len(X_preprocessed_df["S"])-1, n_xlabels, dtype=int)
    plt.xticks(ticks=ticks, labels=X_preprocessed_df["M_TIMESTAMP"].iloc[ticks], rotation=45, fontsize=20)
    plt.xlim((0, len(X_preprocessed_df)))
    plt.xlabel("Date", fontsize=25)
    
    plt.legend(fontsize=20, loc="lower left")
    
    fig.tight_layout()
    plt.show()