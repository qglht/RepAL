import warnings
import os

from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import pipeline

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
    print(f"number of devices : {num_gpus}")

    # create a folder for each group in config['groups'] under model folder
    for group in config["groups"]:
        if not os.path.exists(f"models/{group}"):
            os.makedirs(f"models/{group}")

    for group in config["groups"]:
        for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
            for activation in config["rnn"]["parameters"]["activations"]:
                for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                    for lr in config["rnn"]["parameters"]["learning_rate"]:
                        for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                            device = devices[
                                i % len(devices)
                            ]  # Cycle through available devices
                            tasks.append((group, rnn_type, activation, hidden_size, lr, batch_size, device))
                            i += 1

    # Create a process for each task
    processes = [
        multiprocessing.Process(target=pipeline, args=task) for task in tasks
    ]

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()