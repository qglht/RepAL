import warnings
import os
import argparse

from arrow import get
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import pipeline_mamba, pipeline

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"

if __name__ == "__main__":
    # pipeline("PDM", "pretrain_frozen", "leaky_rnn", "relu", 256, 0.001, 256, "cpu")
    pipeline_mamba("PDM", "master", 16, 1, 1, True, 0.01, 16, "cpu")
