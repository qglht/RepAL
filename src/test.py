import warnings
import os
import argparse

from arrow import get
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import pipeline_mamba, pipeline
from src.toolkit import get_dynamics_mamba, get_dynamics_rnn

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"

if __name__ == "__main__":
    # pipeline("GoNogo", "untrained", "leaky_gru", "leaky_relu", 128, 0.001, 128, "cpu")
    get_dynamics_rnn(
        "leaky_gru",
        "leaky_relu",
        128,
        0.01,
        128,
        "leaky_gru_leaky_relu_128_0.01_128_train.pth",
        "master",
        "PDM",
        device="cpu",
        n_components=20,
    )
    # pipeline_mamba("PDM", "master", 16, 1, 1, True, 0.01, 16, "cpu")
    # get_dynamics_mamba(
    #     16,
    #     1,
    #     0.01,
    #     16,
    #     "mamba_16_1_0.01_16_train.pth",
    #     "master",
    #     "PDM",
    #     "cpu",
    #     n_components=3,
    # )
    # get_dynamics_rnn(
    #     "leaky_gru",
    #     "relu",
    #     256,
    #     0.001,
    #     128,
    #     "leaky_gru_relu_256_0.001_128_train.pth",
    #     "master",
    #     "PDM",
    #     "cpu",
    #     n_components=20,
    # )
