import warnings
import os
import argparse
import torch
import pandas as pd
from dsa_analysis import load_config
from src.toolkit import dissimilarity_over_learning

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'

def dissimilarity_task(params):
    args, rnn_type, activation, hidden_size, lr, batch_size, device = params
    dissimilarities_model = dissimilarity_over_learning(args.group1, args.group2, rnn_type, activation, hidden_size, lr, batch_size, device)
    dissimilarities = {
        "group1": args.group1,
        "group2": args.group2,
        "rnn_type": rnn_type,
        "activation": activation,
        "hidden_size": hidden_size,
        "lr": lr,
        "batch_size": batch_size,
        "cka": dissimilarities_model["cka"],
        "procrustes": dissimilarities_model["procrustes"],
        "dsa": dissimilarities_model["dsa"]
    }
    print(f"dissimilarities dsa : {dissimilarities_model['dsa']}")
    print(f"Len of dissimilarities : dsa {len(dissimilarities_model['dsa'])}")
    return dissimilarities

def dissimilarity(args: argparse.Namespace) -> None:
    config = load_config("config.yaml")
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)] if num_gpus > 0 else [torch.device("cpu")]
    print(f"Number of GPUs available: {num_gpus}")
    device_index = 0
    results = []
    
    diss_index = 0
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        print(f"Index : {100*diss_index/64}")
                        results.append(dissimilarity_task((args, rnn_type, activation, hidden_size, lr, batch_size, devices[device_index])))
    
    # Create DataFrame keeping lists intact
    dissimilarities_df = pd.DataFrame(results)
    dissimilarities_df.to_csv(f"data/dissimilarities_over_learning/{args.group1}_{args.group2}.csv", index=False)

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

    dissimilarity(args)
