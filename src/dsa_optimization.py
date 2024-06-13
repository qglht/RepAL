import warnings
import os

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

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    config = load_config("config.yaml")

    # Create a list of all tasks to run
    tasks = []
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )
    i = 0
    print(f"devices used : {devices}")

    rank=50
    number_parameters_delays = 10
    number_parameters_intervals = 5

    n_delays = np.linspace(5, 50, number_parameters_delays, dtype=int)

    for delay in n_delays:
        delay_interval = np.linspace(1, int(200/delay), number_parameters_intervals, dtype=int)
        for space in delay_interval:
            device = devices[
                i % len(devices)
            ]  # Cycle through available devices
            tasks.append((int(rank), int(delay), int(space), device))
            i += 1


    # Create a process for each task
    processes = [
        multiprocessing.Process(target=dsa_optimisation_compositionality, args=task) for task in tasks
    ]


    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
