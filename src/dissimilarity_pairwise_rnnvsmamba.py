import warnings
import os
import argparse

from matplotlib.pylab import f
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import get_dynamics_rnn, get_dynamics_mamba
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


def measure_dissimilarities(
    model_rnn, model_dict_rnn, model_mamba, model_dict_mamba, groups, taskset, device
):
    config = load_config("config.yaml")
    cka_measure = similarity.make("measure.sim_metric.cka-angular-score")
    procrustes_measure = similarity.make("measure.netrep.procrustes-angular-score")
    curves_names_rnn = list(model_dict_rnn.keys())
    curves_rnn = list(model_dict_rnn.values())
    curves_names_mamba = list(model_dict_mamba.keys())
    curves_mamba = list(model_dict_mamba.values())
    dis_cka = np.zeros((len(groups), len(groups)))
    dis_procrustes = np.zeros((len(groups), len(groups)))
    dis_dsa = np.zeros((len(groups), len(groups)))
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            if groups[i] in curves_names_rnn and groups[j] in curves_names_mamba:
                curve_i = curves_rnn[curves_names_rnn.index(groups[i])]
                curve_j = curves_mamba[curves_names_mamba.index(groups[j])]
                # compute PCA on common basis for 2 groups
                curves, _ = main.compute_common_pca([curve_i, curve_j], n_components=20)
                curve_i = curves[0]
                curve_j = curves[1]
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
    base_dir = f"data/dissimilarities_mamba_rnn/{taskset}"
    measures = ["cka", "procrustes", "dsa"]

    for measure in measures:
        dir_path = os.path.join(base_dir, measure)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it does not exist

        npz_filename = f"{model_rnn.replace('.pth','')}_{model_mamba.replace('.pth','')}.npz"  # Construct filename
        npz_filepath = os.path.join(dir_path, npz_filename)

        np.savez_compressed(npz_filepath, dissimilarities_model[measure])
    return dis_cka, dis_procrustes, dis_dsa


def dissimilarity(args: argparse.Namespace) -> None:
    multiprocessing.set_start_method("spawn", force=True)
    config = load_config("config.yaml")
    groups = [
        "untrained",
        "master",
        "pretrain_basic_anti_frozen",
        "pretrain_frozen",
        "pretrain_unfrozen",
    ]
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )

    curves_rnn = {}
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        model = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}_train.pth"
                        curves_rnn[model] = {}
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
                                curves_rnn[model][group] = copy.deepcopy(curve)

    curves_mamba = {}
    for d_model in config["mamba"]["parameters"]["d_model"]:
        for n_layers in config["mamba"]["parameters"]["n_layers"]:
            for learning_rate in config["mamba"]["parameters"]["learning_rate"]:
                for batch_size in config["mamba"]["parameters"]["batch_size_train"]:
                    model = f"mamba_{d_model}_{n_layers}_{learning_rate}_{batch_size}_train.pth"
                    curves_mamba[model] = {}
                    for group in groups:
                        # check if the model is already trained
                        if os.path.exists(
                            f"models/mamba/{args.taskset}/{group}/{model}"
                        ):
                            print(f"Computing dynamics for {model} and group {group}")
                            curve = get_dynamics_mamba(
                                d_model,
                                n_layers,
                                learning_rate,
                                batch_size,
                                model,
                                group,
                                args.taskset,
                                devices[0],
                            )
                            curves_mamba[model][group] = copy.deepcopy(curve)

    sys.stdout.flush()
    tasks = []
    i = 0
    computed_pairs = set()
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        model_rnn = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}_train.pth"
                        for d_model in config["mamba"]["parameters"]["d_model"]:
                            for n_layers in config["mamba"]["parameters"]["n_layers"]:
                                for learning_rate in config["mamba"]["parameters"][
                                    "learning_rate"
                                ]:
                                    for batch_size in config["mamba"]["parameters"][
                                        "batch_size_train"
                                    ]:
                                        model_mamba = f"mamba_{d_model}_{n_layers}_{learning_rate}_{batch_size}_train.pth"
                                        # check if (model_rnn, model_mamba) is already computed
                                        pair = tuple(sorted((model_rnn, model_mamba)))
                                        if pair not in computed_pairs:
                                            print(
                                                f"Compute dissimilarities for {model_rnn} and {model_mamba}"
                                            )
                                            device = devices[i % len(devices)]
                                            tasks.append(
                                                (
                                                    model_rnn,
                                                    curves_rnn[model],
                                                    model_mamba,
                                                    curves_mamba[model_mamba],
                                                    groups,
                                                    args.taskset,
                                                    device,
                                                )
                                            )
                                            i += 1
                                            computed_pairs.add(pair)

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
