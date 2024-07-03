import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import dissimilarity_within_learning

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"

if __name__ == "__main__":
    # pipeline("gonogo", "leaky_rnn", "leaky_relu", 128, 0.0001, 128, "cpu")
    dissimilarity_within_learning(
        "pretrain_frozen", "leaky_rnn", "leaky_relu", 128, 0.0001, 128, "cpu"
    )
