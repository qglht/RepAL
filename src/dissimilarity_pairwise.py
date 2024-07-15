import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import get_dynamics_model
import numpy as np
import pandas as pd
import similarity
import DSA

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def measure_dissimilarities(group1, group2, model_names_1, model_names_2, device):
    config = load_config("config.yaml")
    cka_measure = similarity.make("measure.sim_metric.cka-angular-score")
    procrustes_measure = similarity.make("measure.netrep.procrustes-angular-score")
    dis_cka = []
    dis_procrustes = []
    dis_dsa = []
    for i, model1 in enumerate(group1):
        for j, model2 in enumerate(group2):
            if model_names_1[i] == model_names_2[j]:
                dis_cka.append(1 - cka_measure(model1, model2))
                dis_procrustes.append(1 - procrustes_measure(model1, model2))
                dsa_comp = DSA.DSA(
                    model1,
                    model2,
                    n_delays=config["dsa"]["n_delays"],
                    rank=config["dsa"]["rank"],
                    delay_interval=config["dsa"]["delay_interval"],
                    verbose=True,
                    iters=1000,
                    lr=1e-2,
                    device=device,
                )
                dis_dsa.append(dsa_comp.fit_score())
    return {"cka": dis_cka, "procrustes": dis_procrustes, "dsa": dis_dsa}


def parse_model_info(model_name):
    model_name = model_name.replace(".pth", "")
    model_name = model_name.split("_")
    model_type = model_name[0] + "_" + model_name[1]
    if len(model_name) == 8:
        activation = model_name[2] + "_" + model_name[3]
        hidden_size = int(model_name[4])
        learning_rate = float(model_name[5])
        batch_size = int(model_name[6])
    else:
        activation = model_name[2]
        hidden_size = int(model_name[3])
        learning_rate = float(model_name[4])
        batch_size = int(model_name[5])
    return model_type, activation, hidden_size, learning_rate, batch_size


def dissimilarity(args: argparse.Namespace) -> None:

    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )

    curves = {group: [] for group in [args.group1, args.group2]}
    explained_variances = {group: [] for group in [args.group1, args.group2]}
    curves_names = {group: [] for group in [args.group1, args.group2]}
    for group in curves.keys():
        for model in os.listdir(f"models/{args.taskset}/{group}"):
            if not model.endswith("_train.pth"):
                continue
            else:
                model_type, activation, hidden_size, lr, batch_size = parse_model_info(
                    model
                )
                print(f"Computing dynamics for {model} and group {group}")
                curve, explained_variance = get_dynamics_model(
                    model_type,
                    activation,
                    hidden_size,
                    lr,
                    model,
                    group,
                    args.taskset,
                    devices[0],
                    n_components=15,
                )
                curves[group].append(curve)
                explained_variances[group].append(explained_variance)
                curves_names[group].append(model.replace(".pth", ""))

    print(f"Dynamics computed")
    dissimilarities = measure_dissimilarities(
        curves[args.group1],
        curves[args.group2],
        curves_names[args.group1],
        curves_names[args.group2],
        devices[0],
    )
    print(f"Dissimilarities computed")
    rows = []

    for i, model1 in enumerate(curves_names[args.group1]):
        # Collect the row data in a dictionary
        print(model1)
        print(i)
        row = {
            "model1": model1,
            "model2": model1,
            "group1": args.group1,
            "group2": args.group2,
            "cka": dissimilarities["cka"][i],
            "procrustes": dissimilarities["procrustes"][i],
            "dsa": dissimilarities["dsa"][i],
            "explained_variance_group1": explained_variances[args.group1][i],
            "explained_variance_group2": explained_variances[args.group2][i],
        }
        # Append the row dictionary to the list
        rows.append(row)

    # Create the DataFrame from the list of rows
    dissimilarities_df = pd.DataFrame(rows)
    dissimilarities_df.to_csv(
        f"data/dissimilarities/{args.taskset}/{args.group1}_{args.group2}.csv",
        index=False,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--taskset",
        type=str,
        default="PDM",
        help="taskset to compare dissimilarities on",
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
