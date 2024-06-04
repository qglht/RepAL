from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import train_model
import ipdb

if __name__ == "__main__":
   
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
    print(f'devices used : {devices}')

    i = 0  # Index to cycle through available devices

    for activation in config["rnn"]["parameters"]["activations"]:
        for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
            for lr in config["rnn"]["parameters"]["learning_rate"]:
                device = devices[i % len(devices)]  # Cycle through available devices
                tasks.append((activation, hidden_size, lr, False, "pretrain", False, device))
                i += 1

    processes = [
        multiprocessing.Process(target=train_model, args=task) for task in tasks
    ]
    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()