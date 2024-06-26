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
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# read all the csv files in data/dsa_restults

if __name__ == "__main__":
    path = "data/dissimilarities_over_learning"
    files = os.listdir(path)
    data = []
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            data.append(df)
    df = pd.concat(data)

    # Assuming your DataFrame is named 'df'
    # If not, replace 'df' with your actual DataFrame name

    # Common percentage points for interpolation
    common_percent = np.linspace(0, 100, 5)

    # Function to create a unique identifier for each model
    def create_model_id(row):
        return f"{row['group1']}_{row['group2']}_{row['rnn_type']}_{row['activation']}_{row['hidden_size']}_{row['lr']}_{row['batch_size']}"

    # Add a model_id column to the DataFrame
    df['model_id'] = df.apply(create_model_id, axis=1)

    # List of dissimilarity measures
    measures = ['cka', 'procrustes', 'dsa']

    # Create a plot for each measure
    for measure in measures:
        plt.figure(figsize=(12, 8))
        
        for _, row in df.iterrows():
            model_id = row['model_id']
            dissimilarities = row[measure]
            
            # Calculate the percentage points for this model
            percent = np.linspace(0, 100, len(dissimilarities))
            
            # Interpolate
            f = interpolate.interp1d(percent, dissimilarities)
            interpolated_dissimilarity = f(common_percent)
            
            # Plot
            plt.plot(common_percent, interpolated_dissimilarity, label=model_id)
        
        plt.xlabel('% of Training')
        plt.ylabel(f'{measure.upper()} Dissimilarity')
        plt.title(f'{measure.upper()} Dissimilarity vs % of Training for Different Models')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()