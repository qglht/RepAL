import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import multiprocessing
import pandas as pd
from src.toolkit import dissimilarity_over_learning

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'

def dissimilarity(args: argparse.Namespace) -> None:
    config = load_config("config.yaml")

    dissimilarities = {"group1":[], "group2":[], "rnn_type":[], "activation":[], "hidden_size":[], "lr":[], "batch_size":[], "cka":[], "procrustes":[], "dsa":[]}
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        device = "cpu"
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
    dissimilarity(args)