import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import multiprocessing
from multiprocessing import Semaphore
from src.toolkit import pipeline_mamba

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def worker(semaphore, task):
    with semaphore:
        pipeline_mamba(*task)


def train(args: argparse.Namespace) -> None:
    multiprocessing.set_start_method("spawn", force=True)
    config = load_config("config.yaml")

    tasks = []
    num_gpus = torch.cuda.device_count()
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )
    i = 0
    print(f"devices used : {devices}")
    print(f"number of devices : {num_gpus}")

    if not os.path.exists(f"models/mamba/{args.taskset}/{args.group}"):
        os.makedirs(f"models/mamba/{args.taskset}/{args.group}")

    for d_model in config["mamba"]["parameters"]["d_model"]:
        for n_layers in config["mamba"]["parameters"]["n_layers"]:
            for learning_rate in config["mamba"]["parameters"]["learning_rate"]:
                for batch_size in config["mamba"]["parameters"]["batch_size_train"]:
                    device = devices[i % len(devices)]
                    tasks.append(
                        (
                            args.taskset,
                            args.group,
                            d_model,
                            n_layers,
                            1,
                            True,
                            learning_rate,
                            batch_size,
                            device,
                        )
                    )
                    i += 1

    semaphore = Semaphore(num_gpus)  # Adjust this number as necessary

    processes = [
        multiprocessing.Process(target=worker, args=(semaphore, task)) for task in tasks
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--group",
        type=str,
        default="master",
        help="The group to train the model on",
    )
    parser.add_argument(
        "--taskset",
        type=str,
        default="PDM",
        help="The tasket to train the model on",
    )
    args = parser.parse_args()
    train(args)
