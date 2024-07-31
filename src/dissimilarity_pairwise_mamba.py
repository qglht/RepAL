import warnings
import os
import argparse

from matplotlib.pylab import f
from sklearn.model_selection import learning_curve
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import get_dynamics_mamba
import numpy as np
import pandas as pd
import similarity
import DSA

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def find_accuracy_model(name, device):
    # Find the latest checkpoint file
    print(f"Finding accuracy for {name}")
    checkpoint_dir = name
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("epoch_") and f.endswith("_checkpoint.pth")
    ]
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1]))
    print(f"Checkpoint files : {checkpoint_files}")

    if checkpoint_files:
        last_checkpoint = checkpoint_files[-1]
        # Load the checkpoint file
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, last_checkpoint), map_location=device
        )

        return float(checkpoint["log"]["perf_min"][-1])
    else:  # return torch nan
        return float(-1)


def measure_dissimilarities(
    group1, group2, model_names_1, model_names_2, accuracy_1, accuracy_2, device
):
    config = load_config("config.yaml")
    cka_measure = similarity.make("measure.sim_metric.cka-angular-score")
    procrustes_measure = similarity.make("measure.netrep.procrustes-angular-score")
    dis_cka = []
    dis_procrustes = []
    dis_dsa = []
    models_selected = []
    accuracies_group1 = []
    accuracies_group2 = []
    for i, model1 in enumerate(group1):
        for j, model2 in enumerate(group2):
            if model_names_1[i] == model_names_2[j]:
                models_selected.append(model_names_1[i])
                accuracies_group1.append(accuracy_1[i])
                accuracies_group2.append(accuracy_2[j])
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
                
    return (
        models_selected,
        accuracies_group1,
        accuracies_group2,
        {
            "cka": dis_cka,
            "procrustes": dis_procrustes,
            "dsa": dis_dsa,
        },
    )


# TODO : return only the models selected


def parse_model_info(model_name):
    model_name = model_name.replace(".pth", "")
    model_name = model_name.split("_")
    d_model = int(model_name[1])
    n_layers = int(model_name[2])
    learning_rate = float(model_name[3])
    batch_size = int(model_name[4])
    return d_model, n_layers, learning_rate, batch_size


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
    accuracies = {group: [] for group in [args.group1, args.group2]}
    for group in curves.keys():
        for model in os.listdir(f"models/mamba/{args.taskset}/{group}"):
            if not model.endswith("_train.pth"):
                continue
            else:
                model_path = os.path.join(
                    f"models/mamba/{args.taskset}/{group}", model.replace(".pth", "")
                )
                d_model, n_layers, learning_rate, batch_size = parse_model_info(model)
                print(f"Computing dynamics for {model} and group {group}")
                curve, explained_variance = get_dynamics_mamba(
                    d_model,
                    n_layers,
                    learning_rate,
                    batch_size,
                    model,
                    group,
                    args.taskset,
                    devices[0],
                )
                final_accuracy = find_accuracy_model(model_path, devices[0])
                curves[group].append(curve)
                explained_variances[group].append(explained_variance)
                curves_names[group].append(model.replace(".pth", ""))
                accuracies[group].append(final_accuracy)

    print(f"Dynamics computed")
    models_selected, accuracies_1, accuracies_2, dissimilarities = (
        measure_dissimilarities(
            curves[args.group1],
            curves[args.group2],
            curves_names[args.group1],
            curves_names[args.group2],
            accuracies[args.group1],
            accuracies[args.group2],
            devices[0],
        )
    )
    print(f"Dissimilarities computed")
    rows = []

    print(f"len of curves_names[args.group1] : {len(curves_names[args.group1])}")
    print(f"len of curves_names[args.group2] : {len(curves_names[args.group2])}")
    print(f"len of dissimilarities['cka'] : {len(dissimilarities['cka'])}")
    print(
        f"len of dissimilarities['procrustes'] : {len(dissimilarities['procrustes'])}"
    )
    print(f"len of dissimilarities['dsa'] : {len(dissimilarities['dsa'])}")
    for i in range(len(models_selected)):
        row = {
            "model1": models_selected[i],
            "model2": models_selected[i],
            "group1": args.group1,
            "group2": args.group2,
            "cka": dissimilarities["cka"][i],
            "procrustes": dissimilarities["procrustes"][i],
            "dsa": dissimilarities["dsa"][i],
            "accuracy_group1": accuracies_1[i],
            "accuracy_group2": accuracies_2[i],
            "explained_variance_group1": explained_variances[args.group1][i],
            "explained_variance_group2": explained_variances[args.group2][i],
        }
        # Append the row dictionary to the list
        rows.append(row)

    # Create the DataFrame from the list of rows
    dissimilarities_df = pd.DataFrame(rows)
    dissimilarities_df.to_csv(
        f"data/dissimilarities/mamba/{args.taskset}/{args.group1}_{args.group2}.csv",
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
