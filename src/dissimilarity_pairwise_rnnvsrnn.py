import warnings
import os
import argparse
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import get_dynamics_rnn
import numpy as np
import similarity
import DSA
import copy
import main
import numpy as np
import sys

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
    model_rnn1,
    model_dict_rnn1,
    accuracies_dict_rnn1,
    model_rnn2,
    model_dict_rnn2,
    accuracies_dict_rnn2,
    groups,
    taskset,
    device,
):

    curves_names_rnn1 = list(model_dict_rnn1.keys())
    curves_rnn1 = list(model_dict_rnn1.values())
    curves_names_rnn2 = list(model_dict_rnn2.keys())
    curves_rnn2 = list(model_dict_rnn2.values())
    dis_cka = np.zeros((len(groups), len(groups)))
    dis_procrustes = np.zeros((len(groups), len(groups)))
    dis_dsa = np.zeros((len(groups), len(groups)))
    accuracies_array_rnn1 = np.zeros((len(groups), 1))
    accuracies_array_rnn2 = np.zeros((len(groups), 1))

    # getting accuracies
    for i in range(len(groups)):
        if groups[i] in accuracies_dict_rnn1:
            accuracies_array_rnn1[i] = accuracies_dict_rnn1[groups[i]]
        if groups[i] in accuracies_dict_rnn2:
            accuracies_array_rnn2[i] = accuracies_dict_rnn2[groups[i]]

    # dissimilarities to compute : master against everything /
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            if groups[i] in curves_names_rnn1 and groups[j] in curves_names_rnn2:
                curve_i = curves_rnn1[curves_names_rnn1.index(groups[i])]
                curve_j = curves_rnn2[curves_names_rnn2.index(groups[j])]
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
        "accuracy_rnn1": accuracies_array_rnn1,
        "accuracy_rnn2": accuracies_array_rnn2,
    }
    base_dir = f"data/dissimilarities_rnn_rnn/{taskset}"
    measures = ["cka", "procrustes", "dsa", "accuracy_rnn1", "accuracy_rnn2"]
    print(f"Saving dissimilarities for {model_rnn1} and {model_rnn2}")
    for measure in measures:
        dir_path = os.path.join(base_dir, measure)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it does not exist

        npz_filename = f"{model_rnn1.replace('.pth','')}_{model_rnn2.replace('.pth','')}.npz"  # Construct filename
        npz_filepath = os.path.join(dir_path, npz_filename)
        print(
            f"Saving dissimilarities for {model_rnn1} and {model_rnn2} in {npz_filepath}"
        )
        np.savez_compressed(npz_filepath, dissimilarities_model[measure])
        print(f"Dissimilarities saved in {npz_filepath}")

    return


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
    for measure in ["cka", "procrustes", "dsa", "accuracy_rnn1", "accuracy_rnn2"]:
        os.makedirs(
            f"data/dissimilarities_rnn_rnn/{args.taskset}/{measure}", exist_ok=True
        )
    curves = {}
    accuracies = {}
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        model = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}_train.pth"
                        model_folder = model.replace(".pth", "")
                        curves[model] = {}
                        accuracies[model] = {}
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
                                accuracies[model][group] = (
                                    find_accuracy_model(
                                        f"models/{args.taskset}/{group}/{model_folder}",
                                        devices[0],
                                    )
                                    if group != "untrained"
                                    else float(-1)
                                )

    sys.stdout.flush()
    computed_pairs = set()
    for rnn_type1 in config["rnn"]["parameters"]["rnn_type"]:
        for activation1 in config["rnn"]["parameters"]["activations"]:
            for hidden_size1 in config["rnn"]["parameters"]["n_rnn"]:
                for lr1 in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size1 in config["rnn"]["parameters"]["batch_size_train"]:
                        model1 = f"{rnn_type1}_{activation1}_{hidden_size1}_{lr1}_{batch_size1}_train.pth"
                        for rnn_type2 in config["rnn"]["parameters"]["rnn_type"]:
                            for activation2 in config["rnn"]["parameters"][
                                "activations"
                            ]:
                                for hidden_size2 in config["rnn"]["parameters"][
                                    "n_rnn"
                                ]:
                                    for lr2 in config["rnn"]["parameters"][
                                        "learning_rate"
                                    ]:
                                        for batch_size2 in config["rnn"]["parameters"][
                                            "batch_size_train"
                                        ]:
                                            model2 = f"{rnn_type2}_{activation2}_{hidden_size2}_{lr2}_{batch_size2}_train.pth"
                                            device = devices[0]
                                            pair = sorted([model1, model2])
                                            if pair not in computed_pairs:
                                                print(
                                                    f"Compute dissimilarities for {mode1} and {model2}"
                                                )
                                                measure_dissimilarities(
                                                    model1,
                                                    curves[model1],
                                                    accuracies[model1],
                                                    model2,
                                                    curves[model2],
                                                    accuracies[model2],
                                                    groups,
                                                    args.taskset,
                                                    device,
                                                )
                                                computed_pairs.add(pair)

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
