import warnings
import os
import argparse
import torch
import numpy as np
from dsa_analysis import load_config
from src.toolkit import dissimilarity_over_learning_mamba
import multiprocessing

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def dissimilarity_task(
    taskset,
    group1,
    group2,
    d_model,
    n_layers,
    learning_rate,
    batch_size,
    device,
):
    dissimilarities_model = dissimilarity_over_learning_mamba(
        taskset,
        group1,
        group2,
        d_model,
        n_layers,
        learning_rate,
        batch_size,
        device,
    )

    base_dir = f"data/dissimilarities_over_learning/mamba/{taskset}"
    measures = ["cka", "procrustes", "dsa", "accuracy_1", "accuracy_2"]

    for measure in measures:
        dir_path = os.path.join(base_dir, f"{group1}_{group2}", measure)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it does not exist

        npz_filename = f"mamba_{d_model}_{n_layers}_{learning_rate}_{batch_size}.npz"  # Construct filename
        npz_filepath = os.path.join(dir_path, npz_filename)

        np.savez_compressed(npz_filepath, dissimilarities_model[measure])
    return dissimilarities_model


def dissimilarity(args: argparse.Namespace) -> None:
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
    print(f"number of devices : {num_gpus}")

    if not os.path.exists(
        f"data/dissimilarities_over_learning/mamba/{args.taskset}/{args.group1}_{args.group2}"
    ):
        os.makedirs(
            f"data/dissimilarities_over_learning/mamba/{args.taskset}/{args.group1}_{args.group2}",
            exist_ok=True,
        )

    for d_model in config["mamba"]["parameters"]["d_model"]:
        for n_layers in config["mamba"]["parameters"]["n_layers"]:
            for learning_rate in config["mamba"]["parameters"]["learning_rate"]:
                for batch_size in config["mamba"]["parameters"]["batch_size_train"]:
                    device = devices[
                        i % len(devices)
                    ]  # Cycle through available devices
                    tasks.append(
                        (
                            args.taskset,
                            args.group1,
                            args.group2,
                            d_model,
                            n_layers,
                            learning_rate,
                            batch_size,
                            device,
                        )
                    )
                    i += 1

    # Create a process for each task
    processes = [
        multiprocessing.Process(target=dissimilarity_task, args=task) for task in tasks
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
        help="taskset to compare on",
    )
    parser.add_argument(
        "--group1",
        type=str,
        default="pretrain_frozen",
        help="group to compare 1",
    )
    parser.add_argument(
        "--group2",
        type=str,
        default="pretrain_unfrozen",
        help="group to compare 2",
    )
    args = parser.parse_args()

    dissimilarity(args)
