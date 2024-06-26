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
    path = "data/dissimilarities"
    files = os.listdir(path)
    data = []
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            data.append(df)
    data = pd.concat(data)
    data_groupes_mean_dissimilarity = data.groupby(['group1', 'group2'])[["cka","procrustes","dsa"]].mean().reset_index()
    data_models = data[data['model1'] == data['model2']].groupby(['group1', 'group2','model1','model2'])[["cka","procrustes","dsa"]].mean().reset_index()
    data_models_averaged = data_models.groupby(['group1', 'group2'])[["cka","procrustes","dsa"]].mean().reset_index()
    
    # plot heatmap of the dissimilarity between the groups
    df = pd.DataFrame(data_models_averaged)

    # Pivot the dataframe to prepare for heatmap
    pivot_df = df.pivot(index='group1', columns='group2', values=['cka', 'procrustes', 'dsa'])

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df['cka'], annot=True, cmap='coolwarm', cbar=True, vmin=0, vmax=1, square=True)
    plt.title('CKA Dissimilarity')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df['procrustes'], annot=True, cmap='coolwarm', cbar=True, vmin=0, vmax=1, square=True)
    plt.title('Procrustes Dissimilarity')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df['dsa'], annot=True, cmap='coolwarm', cbar=True, vmin=0, vmax=1, square=True)
    plt.title('DSA Dissimilarity')
    plt.show()
    