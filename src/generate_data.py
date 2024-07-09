import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import generate_data

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def generate(args: argparse.Namespace) -> None:

    multiprocessing.set_start_method(
        "spawn", force=True
    )  # Set multiprocessing to use 'spawn'
    config = load_config("config.yaml")

    # Create a list of all tasks to run
    tasks = []
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )
    print(f"devices used : {devices}")

    i = 0  # Index to cycle through available devices

    for env in config[args.taskset]["all_rules"]:
        device = devices[i % len(devices)]  # Cycle through available devices
        tasks.append((args.taskset, env))
        i += 1

    print([task for task in tasks])
    processes = [
        multiprocessing.Process(target=generate_data, args=(task,)) for task in tasks
    ]
    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--taskset",
        type=str,
        default="PDM",
        help="The taskset to train the model on",
    )
    args = parser.parse_args()
    generate(args)
