import numpy as np
import os
import sys
from subprocess import call
import warnings
import argparse
import seaborn as sns
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import pipeline
from src.dsa_optimization import dsa_computation
import pandas as pd
import matplotlib.pyplot as plt
import ipdb
import matplotlib.colors as mcolors

# read all the csv files in data/dsa_restults

if __name__ == "__main__":
    path = "data/dsa_results"
    files = os.listdir(path)
    data = []
    for file in files:
        if file.endswith(".csv"):
            if not file.endswith("ordered.csv"):
                # split the name of the file to get the parameters
                parameters = file.split("_")
                n_delay = int(parameters[1])
                delay_interval = int(parameters[2].split(".")[0])
                df = pd.read_csv(os.path.join(path, file))
                df["n_delay"] = n_delay
                df["delay_interval"] = delay_interval
                data.append(df)
    data = pd.concat(data)
    ipdb.set_trace()
    data.rename(columns={"number of shared elements":"number_of_shared_elements"}, inplace=True)
    # remove Nan values
    data = data.dropna()# plot on a single plot the Median similarity vs Number of shared elements for pair  (n_delay, delay_interval)
    # ipdb.set_trace()
    # Group by `n_delay` and `delay_interval`
    data = data.groupby(['n_delay', 'delay_interval','number_of_shared_elements'])["similarity"].mean().reset_index()
    groups = data.groupby(['n_delay', 'delay_interval'])

    # Plot 1: Influence of n_delay
    plt.figure(figsize=(14, 8))

    # Generate a consistent color scale for n_delay
    base_colors_n_delay = list(mcolors.TABLEAU_COLORS.values())
    color_map_n_delay = {}

    for i, n_delay in enumerate(data['n_delay'].unique()):
        shades = np.linspace(0.3, 1, len(data['delay_interval'].unique()))
        for j, delay_interval in enumerate(data['delay_interval'].unique()):
            color_map_n_delay[(n_delay, delay_interval)] = mcolors.to_rgba(base_colors_n_delay[i % len(base_colors_n_delay)], shades[j])

    # Plotting
    for (n_delay, delay_interval), group in groups:
        color = color_map_n_delay[(n_delay, delay_interval)]
        plt.plot(group['number_of_shared_elements'], group['similarity'], 
                label=f'n_delay={n_delay}, delay_interval={delay_interval}', 
                color=color, alpha=0.8)

    # Adding labels and title
    plt.xlabel('Number of shared elements')
    plt.ylabel('Median similarity')
    plt.title('Influence of n_delay on Median similarity vs Number of shared elements')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Influence of delay_interval
    plt.figure(figsize=(14, 8))

    # Generate a consistent color scale for delay_interval
    base_colors_n_delay = list(mcolors.TABLEAU_COLORS.values())
    color_map_n_delay = {}

    for i, delay_interval in enumerate(data['delay_interval'].unique()):
        shades = np.linspace(0.3, 1, len(data['n_delay'].unique()))
        for j, n_delay in enumerate(data['n_delay'].unique()):
            color_map_n_delay[(n_delay, delay_interval)] = mcolors.to_rgba(base_colors_n_delay[i % len(base_colors_n_delay)], shades[j])

    # Plotting
    for (n_delay, delay_interval), group in groups:
        color = color_map_n_delay[(n_delay, delay_interval)]
        plt.plot(group['number_of_shared_elements'], group['similarity'], 
                label=f'n_delay={n_delay}, delay_interval={delay_interval}', 
                color=color, alpha=0.8)

    # Adding labels and title
    plt.xlabel('Number of shared elements')
    plt.ylabel('Median similarity')
    plt.title('Influence of n_delay on Median similarity vs Number of shared elements')
    plt.legend()
    plt.grid(True)
    plt.show()