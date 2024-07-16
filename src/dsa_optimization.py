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
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def dsa_computation(args: argparse.Namespace) -> None:
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )

    rank = 50

    dsa_optimisation_compositionality(
        rank, args.n_delay, args.delay_interval, devices[0], args.ordered
    )
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
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="if taking into account order or not",
    )
    parser.add_argument(
        "--no-ordered",
        action="store_false",
        dest="ordered",
        help="if not taking into account order",
    )
    parser.set_defaults(ordered=False)
    args = parser.parse_args()
    dsa_computation(args)
