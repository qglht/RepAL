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
    ipdb.set_trace()
    