import warnings
import os
import argparse
from matplotlib.pylab import f
import torch
import multiprocessing
import numpy as np
import copy
import sys
import gc
from dsa_analysis import load_config
from src.toolkit import get_dynamics_rnn, get_dynamics_mamba
import similarity
import DSA
import main

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

    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("_")[1]))
        print(f"Checkpoint files : {checkpoint_files}")
        if len(checkpoint_files) < 50:
            return float(1)
        else:
            last_checkpoint = checkpoint_files[-1]
            # Load the checkpoint file
            checkpoint = torch.load(
                os.path.join(checkpoint_dir, last_checkpoint), map_location=device
            )

            return float(checkpoint["log"]["perf_min"][-1])
    else:  # return torch nan
        return float(-1)


def measure_dissimilarities(
    model_rnn,
    model_dict_rnn,
    accuracies_dict_rnn,
    model_mamba,
    model_dict_mamba,
    accuracies_dict_mamba,
    groups,
    taskset,
    device,
):

    curves_names_rnn = list(model_dict_rnn.keys())
    curves_rnn = list(model_dict_rnn.values())
    curves_names_mamba = list(model_dict_mamba.keys())
    curves_mamba = list(model_dict_mamba.values())
    dis_cka = np.zeros((len(groups), len(groups)))
    dis_procrustes = np.zeros((len(groups), len(groups)))
    dis_dsa = np.zeros((len(groups), len(groups)))
    accuracies_array_rnn = np.zeros((len(groups), 1))
    accuracies_array_mamba = np.zeros((len(groups), 1))

    # getting accuracies
    for i in range(len(groups)):
        if groups[i] in accuracies_dict_rnn:
            accuracies_array_rnn[i] = accuracies_dict_rnn[groups[i]]
        if groups[i] in accuracies_dict_mamba:
            accuracies_array_mamba[i] = accuracies_dict_mamba[groups[i]]

    # dissimilarities to compute : master against everything /
    for i in range(len(groups)):
        for j in range(len(groups)):
            if groups[i] in curves_names_rnn and groups[j] in curves_names_mamba:
                curve_i = curves_rnn[curves_names_rnn.index(groups[i])]
                curve_j = curves_mamba[curves_names_mamba.index(groups[j])]
                # compute PCA on common basis for 2 groups
                curves_pca, _ = main.compute_common_pca(
                    [curve_i, curve_j], n_components=20
                )
                curve_i = curves_pca[0]
                curve_j = curves_pca[1]
                dis_cka[i, j] = 1 - cka_measure(curve_i, curve_j)
                dis_procrustes[i, j] = 1 - procrustes_measure(curve_i, curve_j)
                dsa_computation = DSA.DSA(
                    curve_i,
                    curve_j,
                    n_delays=config["dsa"]["n_delays"],
                    rank=config["dsa"]["rank"],
                    delay_interval=config["dsa"]["delay_interval"],
                    verbose=True,
                    iters=1000,
                    lr=1e-2,
                    device=device,
                )
                dis_dsa[i, j] = dsa_computation.fit_score()
            else:
                dis_cka[i, j] = np.nan
                dis_procrustes[i, j] = np.nan
                dis_dsa[i, j] = np.nan

    dissimilarities_model = {
        "cka": dis_cka,
        "procrustes": dis_procrustes,
        "dsa": dis_dsa,
        "accuracy_rnn": accuracies_array_rnn,
        "accuracy_mamba": accuracies_array_mamba,
    }
    base_dir = f"data/dissimilarities_mamba_rnn/{taskset}"
    measures = ["cka", "procrustes", "dsa", "accuracy_rnn", "accuracy_mamba"]
    print(f"Saving dissimilarities for {model_rnn} and {model_mamba}")
    for measure in measures:
        dir_path = os.path.join(base_dir, measure)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it does not exist

        npz_filename = f"{model_rnn.replace('.pth','')}_{model_mamba.replace('.pth','')}.npz"  # Construct filename
        npz_filepath = os.path.join(dir_path, npz_filename)
        print(
            f"Saving dissimilarities for {model_rnn} and {model_mamba} in {npz_filepath}"
        )
        np.savez_compressed(npz_filepath, dissimilarities_model[measure])
        print(f"Dissimilarities saved in {npz_filepath}")

    return


def dissimilarity(args: argparse.Namespace) -> None:
    groups = [
        "untrained",
        "master_frozen",
        "master",
        "pretrain_basic_frozen",
        "pretrain_anti_frozen",
        "pretrain_delay_frozen",
        "pretrain_basic_anti_frozen",
        "pretrain_basic_delay_frozen",
        "pretrain_frozen",
        "pretrain_unfrozen",
    ]
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )
    for measure in ["cka", "procrustes", "dsa", "accuracy_rnn", "accuracy_mamba"]:
        os.makedirs(
            f"data/dissimilarities_mamba_rnn/{args.taskset}/{measure}", exist_ok=True
        )

    print(f"Computing dynamics for all models")
    curves_rnn = {}
    accuracies_rnn = {}
    rnn_type = "leaky_rnn"
    activation = "relu"
    for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
        for lr in config["rnn"]["parameters"]["learning_rate"]:
            for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                model = (
                    f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}_train.pth"
                )
                model_folder = model.replace(".pth", "")
                curves_rnn[model] = {}
                accuracies_rnn[model] = {}
                for group in groups:
                    # check if the model is already trained
                    if os.path.exists(f"models/{args.taskset}/{group}/{model}"):
                        curves_rnn[model][group] = get_dynamics_rnn(
                            rnn_type,
                            activation,
                            hidden_size,
                            lr,
                            batch_size,
                            model,
                            group,
                            args.taskset,
                            devices[0],
                        )
                        accuracies_rnn[model][group] = (
                            find_accuracy_model(
                                f"models/{args.taskset}/{group}/{model_folder}",
                                devices[0],
                            )
                            if group != "untrained"
                            else float(-1)
                        )

    curves_mamba = {}
    accuracies_mamba = {}
    d_model = 8
    n_layers = 1
    for learning_rate in config["mamba"]["parameters"]["learning_rate"]:
        for batch_size in config["mamba"]["parameters"]["batch_size_train"]:
            model = f"mamba_{d_model}_{n_layers}_{learning_rate}_{batch_size}_train.pth"
            model_folder = model.replace(".pth", "")
            curves_mamba[model] = {}
            accuracies_mamba[model] = {}
            for group in groups:
                # check if the model is already trained
                if os.path.exists(f"models/mamba/{args.taskset}/{group}/{model}"):
                    curves_mamba[model][group] = get_dynamics_mamba(
                        d_model,
                        n_layers,
                        learning_rate,
                        batch_size,
                        model,
                        group,
                        args.taskset,
                        devices[0],
                    )
                    accuracies_mamba[model][group] = (
                        find_accuracy_model(
                            f"models/mamba/{args.taskset}/{group}/{model_folder}",
                            devices[0],
                        )
                        if group != "untrained"
                        else float(-1)
                    )

    sys.stdout.flush()
    print(f"Computing dissimilarities for all models")
    computed_pairs = set()
    rnn_type = "leaky_rnn"
    activation = "relu"
    for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
        for lr in config["rnn"]["parameters"]["learning_rate"]:
            for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                model_rnn = (
                    f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}_train.pth"
                )
                d_model = 8
                n_layers = 1
                for learning_rate in config["mamba"]["parameters"]["learning_rate"]:
                    for batch_size in config["mamba"]["parameters"]["batch_size_train"]:
                        model_mamba = f"mamba_{d_model}_{n_layers}_{learning_rate}_{batch_size}_train.pth"
                        # check if (model_rnn, model_mamba) is already computed
                        pair = tuple(sorted((model_rnn, model_mamba)))
                        if pair not in computed_pairs:
                            print(
                                f"Compute dissimilarities for {model_rnn} and {model_mamba}"
                            )
                            device = devices[0]
                            measure_dissimilarities(
                                model_rnn,
                                curves_rnn[model_rnn],
                                accuracies_rnn[model_rnn],
                                model_mamba,
                                curves_mamba[model_mamba],
                                accuracies_mamba[model_mamba],
                                groups,
                                args.taskset,
                                device,
                            )
                            computed_pairs.add(pair)


if __name__ == "__main__":
    config = load_config("config.yaml")
    cka_measure = similarity.make("measure.sim_metric.cka-angular-score")
    procrustes_measure = similarity.make("measure.netrep.procrustes-angular-score")
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--taskset",
        type=str,
        default="PDM",
        help="taskset to compare dissimilarities on",
    )
    args = parser.parse_args()
    dissimilarity(args)
