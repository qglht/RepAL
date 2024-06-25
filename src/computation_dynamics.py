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
            # split the name of the file to get the parameters
            parameters = file.split("_")
            group1 = parameters[0]
            group2 = parameters[1].split(".")[0]
            df = pd.read_csv(os.path.join(path, file))
            df["group"] = (group1, group2)
            data.append(df)
    data = pd.concat(data)
    