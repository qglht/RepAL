import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import dsa_optimisation_compositionality
import numpy as np
import pandas as pd

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'

def dsa_computation(args: argparse.Namespace) -> None:

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    rank=50

    dsa_optimisation_compositionality(rank, args.n_delay, args.delay_interval, device)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dsa optimisation")
    parser.add_argument(
        "--n_delay",
        type=int,
        default=5,
        help="The number of delays to compute DSA",
    )
    parser.add_argument(
        "--delay_interval",
        type=int,
        default=1,
        help="The delay interval to compute DSA",
    )
    args = parser.parse_args()
    dsa_computation(args)