import warnings
import os
import argparse

from matplotlib.pylab import f
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import get_dynamics_rnn
import numpy as np
import pandas as pd
import similarity
import DSA
import copy
import main
import numpy as np
import sys
import ipdb

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def worker(task):
    try:
        measure_dissimilarities(*task)
    except Exception as e:
        print(f"Error in worker: {e}")


def measure_dissimilarities(model, model_dict, groups, taskset, device):
    config = load_config("config.yaml")
    cka_measure = similarity.make("measure.sim_metric.cka-angular-score")
    procrustes_measure = similarity.make("measure.netrep.procrustes-angular-score")
    curves_names = list(model_dict.keys())
    curves = list(model_dict.values())
    dis_cka = np.zeros((len(groups), len(groups)))
    dis_procrustes = np.zeros((len(groups), len(groups)))
    dis_dsa = np.zeros((len(groups), len(groups)))
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            if groups[i] in curves_names and groups[j] in curves_names:
                curve_i = curves[curves_names.index(groups[i])]
                curve_j = curves[curves_names.index(groups[j])]
                # compute PCA on common basis for 2 groups
                curves_pca, _ = main.compute_common_pca(
                    [curve_i, curve_j], n_components=20
                )
                print(f"Computing dissimilarities between {groups[i]} and {groups[j]}")
                curve_i = curves_pca[0]
                curve_j = curves_pca[1]
                dis_cka[i, j] = 1 - cka_measure(
                    curve_i,
                    curve_j,
                )
                dis_procrustes[i, j] = 1 - procrustes_measure(
                    curve_i,
                    curve_j,
                )
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
    }
    base_dir = f"data/dissimilarities/{taskset}"
    measures = ["cka", "procrustes", "dsa"]

    print(f"Saving dissimilarities for {model}")
    for measure in measures:
        dir_path = os.path.join(base_dir, measure)

        npz_filename = f"{model.replace('.pth','')}.npz"  # Construct filename
        npz_filepath = os.path.join(dir_path, npz_filename)
        print(f"Saving dissimilarities for {model} in {npz_filepath}")
        np.savez_compressed(npz_filepath, dissimilarities_model[measure])
    return dis_cka, dis_procrustes, dis_dsa


def dissimilarity(args: argparse.Namespace) -> None:
    multiprocessing.set_start_method("spawn", force=True)
    config = load_config("config.yaml")
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
    # create directories if don't exist
    for measure in ["cka", "procrustes", "dsa"]:
        os.makedirs(f"data/dissimilarities/{args.taskset}/{measure}", exist_ok=True)
    curves = {}
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        model = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}_train.pth"
                        curves[model] = {}
                        for group in groups:
                            # check if the model is already trained
                            if os.path.exists(f"models/{args.taskset}/{group}/{model}"):
                                print(
                                    f"Computing dynamics for {model} and group {group}"
                                )
                                curve = get_dynamics_rnn(
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
                                curves[model][group] = copy.deepcopy(curve)

    sys.stdout.flush()
    tasks = []
    i = 0
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        model = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}_train.pth"
                        device = devices[
                            i % len(devices)
                        ]  # Cycle through available devices
                        print(f"Compute dissimilarities for {model}")
                        tasks.append(
                            (
                                model,
                                curves[model],
                                groups,
                                args.taskset,
                                device,
                            )
                        )
                        i += 1

    # Create a process for each task
    processes = [multiprocessing.Process(target=worker, args=(task,)) for task in tasks]

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--taskset",
        type=str,
        default="PDM",
        help="taskset to compare dissimilarities on",
    )
    args = parser.parse_args()
    dissimilarity(args)
