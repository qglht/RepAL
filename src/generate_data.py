from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import generate_data
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

    for env in config['rnn']['rules']:
        device = devices[i % len(devices)]  # Cycle through available devices
        tasks.append((env))
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