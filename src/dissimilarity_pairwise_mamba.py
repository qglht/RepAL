import warnings
import os
import argparse

from matplotlib.pylab import f
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import get_dynamics_mamba
import numpy as np
import pandas as pd
import similarity
import DSA
import copy
import main
import numpy as np
import sys
import logging

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def setup_logging(log_dir):
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logging object and set its level
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers in subsequent calls
    if not logger.handlers:
        # Create file handler to write logs to a file
        file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
        file_handler.setLevel(logging.INFO)

        # Create console handler to logging.info logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Define log message format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def worker(task):
    try:
        measure_dissimilarities(*task)
    except Exception as e:
        logging.info(f"Error in worker: {e}")


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
                dis_cka[i, j] = 1 - cka_measure(
                    curves[curves_names.index(groups[i])],
                    curves[curves_names.index(groups[j])],
                )
                dis_procrustes[i, j] = 1 - procrustes_measure(
                    curves[curves_names.index(groups[i])],
                    curves[curves_names.index(groups[j])],
                )
                dsa_computation = DSA.DSA(
                    curves[curves_names.index(groups[i])],
                    curves[curves_names.index(groups[j])],
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
    base_dir = f"data/dissimilarities/mamba/{taskset}"
    measures = ["cka", "procrustes", "dsa"]

    for measure in measures:
        dir_path = os.path.join(base_dir, measure)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it does not exist

        npz_filename = f"{model.replace('.pth','')}.npz"  # Construct filename
        npz_filepath = os.path.join(dir_path, npz_filename)

        np.savez_compressed(npz_filepath, dissimilarities_model[measure])
    return dis_cka, dis_procrustes, dis_dsa


def dissimilarity(args: argparse.Namespace) -> None:
    multiprocessing.set_start_method("spawn", force=True)
    logging = setup_logging(
        os.path.join(f"data/dissimilarities/{args.taskset}", "logs")
    )
    config = load_config("config.yaml")
    groups = [
        "pretrain_frozen",
        "pretrain_unfrozen",
        "pretrained_basic_anti_frozen",
        "pretrained_basic_anti_unfrozen",
        "master",
        "untrained",
        "basic",
        "anti",
    ]
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )

    curves = {}
    for d_model in config["mamba"]["parameters"]["d_model"]:
        for n_layers in config["mamba"]["parameters"]["n_layers"]:
            for learning_rate in config["mamba"]["parameters"]["learning_rate"]:
                for batch_size in config["mamba"]["parameters"]["batch_size_train"]:
                    model = f"mamba_{d_model}_{n_layers}_{learning_rate}_{batch_size}_train.pth"
                    curves[model] = {}
                    for group in groups:
                        # check if the model is already trained
                        if os.path.exists(
                            f"models/mamba/{args.taskset}/{group}/{model}"
                        ):
                            logging.info(
                                f"Computing dynamics for {model} and group {group}"
                            )
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
                            curves[model][group] = copy.deepcopy(curve)
                    # apply PCA on common basis for all groups for the given model
                    logging.info(f"Computing PCA for {model}")
                    curves_group = list(curves[model].keys())
                    curves_reduced, _ = main.compute_common_pca(
                        list(curves[model].values()), n_components=20
                    )
                    # update
                    for group in curves_group:
                        curves[model][group] = curves_reduced[curves_group.index(group)]

    sys.stdout.flush()
    tasks = []
    i = 0
    for d_model in config["mamba"]["parameters"]["d_model"]:
        for n_layers in config["mamba"]["parameters"]["n_layers"]:
            for learning_rate in config["mamba"]["parameters"]["learning_rate"]:
                for batch_size in config["mamba"]["parameters"]["batch_size_train"]:
                    model = f"mamba_{d_model}_{n_layers}_{learning_rate}_{batch_size}_train.pth"
                    logging.info(f"Compute dissimilarities for {model}")
                    device = devices[
                        i % len(devices)
                    ]  # Cycle through available devices
                    tasks.append((model, curves[model], groups, args.taskset, device))
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
