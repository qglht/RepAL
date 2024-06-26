import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import pandas as pd
from src.toolkit import dissimilarity_over_learning
from torch.multiprocessing import Pool, set_start_method

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'

def dissimilarity_task(params):
    args, rnn_type, activation, hidden_size, lr, batch_size, device = params
    dissimilarities = {"group1": [], "group2": [], "rnn_type": [], "activation": [], "hidden_size": [], "lr": [], "batch_size": [], "cka": [], "procrustes": [], "dsa": []}
    dissimilarities_model = dissimilarity_over_learning(args.group1, args.group2, rnn_type, activation, hidden_size, lr, batch_size, device)
    dissimilarities["group1"] = args.group1
    dissimilarities["group2"] = args.group2
    dissimilarities["rnn_type"] = rnn_type
    dissimilarities["activation"] = activation
    dissimilarities["hidden_size"] = hidden_size
    dissimilarities["lr"] = lr
    dissimilarities["batch_size"] = batch_size
    dissimilarities["cka"] = dissimilarities_model["cka"]
    dissimilarities["procrustes"] = dissimilarities_model["procrustes"]
    dissimilarities["dsa"] = dissimilarities_model["dsa"]
    return dissimilarities

def dissimilarity(args: argparse.Namespace) -> None:
    config = load_config("config.yaml")
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)] if num_gpus > 0 else [torch.device("cpu")]
    
    tasks = []
    device_index = 0
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        device = devices[device_index % num_gpus]  # Assign each task to a different GPU
                        tasks.append((args, rnn_type, activation, hidden_size, lr, batch_size, device))
                        device_index += 1
    
    with Pool(processes=num_gpus) as pool:
        results = pool.map(dissimilarity_task, tasks)
    
    dissimilarities = {"group1": [], "group2": [], "rnn_type": [], "activation": [], "hidden_size": [], "lr": [], "batch_size": [], "cka": [], "procrustes": [], "dsa": []}
    for result in results:
        for key in dissimilarities.keys():
            dissimilarities[key].append(result[key])
    
    dissimilarities_df = pd.DataFrame(dissimilarities)
    dissimilarities_df.to_csv(f"dissimilarities_over_learning_{args.group1}_{args.group2}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
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
    
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    dissimilarity(args)

