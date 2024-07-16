import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import concurrent.futures
from src.toolkit import pipeline_mamba

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def train(args: argparse.Namespace) -> None:
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

    # create a folder for each group in config['groups'] under model folder
    if not os.path.exists(f"models/mamba/{args.taskset}/{args.group}"):
        os.makedirs(f"models/mamba/{args.taskset}/{args.group}")

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
                            args.group,
                            d_model,
                            n_layers,
                            1,  # pad_vocab_size_multiple
                            True,  # pscan
                            learning_rate,
                            batch_size,
                            device,
                        )
                    )
                    i += 1

    # Limit the number of concurrent processes
    max_workers = min(8, len(tasks))  # Adjust the number of workers based on your needs

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(pipeline_mamba, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")


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
        help="The taskset to train the model on",
    )
    args = parser.parse_args()
    train(args)
