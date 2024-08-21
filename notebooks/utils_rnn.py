import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from dsa_analysis import load_config, visualize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
import main
from main import RNNLayer
import ast
import DSA
import copy

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

measures = ["cka", "dsa", "procrustes"]


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


def symmetric(array):
    for i in range(array.shape[0]):
        for j in range(i, array.shape[1]):
            array[j, i] = array[i, j]
    return array


# function to replace all nan which are on diagonal of an array
def replace_nan_diagonal(array):
    for i in range(array.shape[0]):
        if np.isnan(array[i, i]):
            array[i, i] = 0
    return array


def replace_nan_diagonal_list(arrays):
    return [replace_nan_diagonal(arr) for arr in arrays]


# function to remove among list of arrays the ones which have nan values within array
def remove_nan(array):
    return array if not np.isnan(np.sum(array)) else None


def remove_nan_list(lists):
    return [remove_nan(arr) for arr in lists]


def get_dataframe(path, taskset):
    path = f"../data/dissimilarities/{taskset}/"
    df = {
        "model_type": [],
        "activation": [],
        "hidden_size": [],
        "lr": [],
        "batch_size": [],
        "group1": [],
        "group2": [],
        "measure": [],
        "dissimilarity": [],
    }

    dissimilarities = {measure: [] for measure in measures}
    for measure in measures:
        path_measure = os.path.join(path, measure)
        files = os.listdir(path_measure)
        for file in files:
            file_path = os.path.join(path_measure, file)
            if file_path.endswith(".npz"):
                model_type, activation, hidden_size, lr, batch_size = parse_model_info(
                    file
                )
                with np.load(file_path, allow_pickle=True) as data:
                    dissimilarities[measure].append(data["arr_0"])
                    for i in range(len(groups)):
                        for j in range(len(groups)):
                            array_dissimilarities = remove_nan(
                                replace_nan_diagonal(data["arr_0"])
                            )
                            if array_dissimilarities is not None:
                                array_dissimilarities = symmetric(array_dissimilarities)
                                df["model_type"].append(model_type)
                                df["activation"].append(activation)
                                df["hidden_size"].append(hidden_size)
                                df["lr"].append(lr)
                                df["batch_size"].append(batch_size)
                                df["group1"].append(groups[i])
                                df["group2"].append(groups[j])
                                df["measure"].append(measure)
                                df["dissimilarity"].append(array_dissimilarities[i, j])
    return pd.DataFrame(df)


def find_group_pairs(config, taskset):
    groups = list(config[taskset]["groups"].keys())
    # generate all possible pairs of groups
    pairs = [
        (groups[i], groups[j])
        for i in range(len(groups))
        for j in range(i, len(groups))
    ]
    # group pairs of groups by how many tasks they share in their training curriculum
    group_pairs = {}
    for pair in pairs:
        group1, group2 = pair
        group1_tasks = (
            config[taskset]["groups"][group1]["pretrain"]["ruleset"]
            # + config[taskset]["groups"][group1]["train"]["ruleset"]
            # if config[taskset]["groups"][group1]["train"]["frozen"] == False
            # else config[taskset]["groups"][group1]["pretrain"]["ruleset"]
        )
        group2_tasks = (
            config[taskset]["groups"][group2]["pretrain"]["ruleset"]
            # + config[taskset]["groups"][group2]["train"]["ruleset"]
            # if config[taskset]["groups"][group2]["train"]["frozen"] == False
            # else config[taskset]["groups"][group2]["pretrain"]["ruleset"]
        )
        if len(group1_tasks) == 0 and len(group2_tasks) == 0:
            shared_tasks = 100
        elif len(group1_tasks) == 0 or len(group2_tasks) == 0:
            shared_tasks = 0
        else:
            shared_tasks = int(
                100
                * len(set(group1_tasks).intersection(set(group2_tasks)))
                / max(len(group1_tasks), len(group2_tasks))
            )
        try:
            group_pairs[shared_tasks].append(pair)
        except KeyError:
            group_pairs[shared_tasks] = [pair]
    return group_pairs


def find_group_pairs_master(config, taskset):
    groups = list(config[taskset]["groups"].keys())
    # generate all possible pairs of groups
    pairs = [
        (groups[i], groups[j])
        for i in range(len(groups))
        for j in range(i, len(groups))
        if groups[i] == "master" or groups[j] == "master"
    ]
    # remove the (master, master) pair
    pairs = [pair for pair in pairs if pair[0] != pair[1]]
    # remove the pair containign unfrozen and the one containing master_frozen
    pairs = [
        pair
        for pair in pairs
        if pair[0] != "pretrain_unfrozen" and pair[1] != "pretrain_unfrozen"
    ]
    pairs = [
        pair
        for pair in pairs
        if pair[0] != "master_frozen" and pair[1] != "master_frozen"
    ]

    # group pairs of groups by how many tasks they share in their training curriculum
    group_pairs = {}
    for pair in pairs:
        group1, group2 = pair
        if group1 == "master":
            group2_tasks = (
                config[taskset]["groups"][group2]["pretrain"]["ruleset"]
                # + config[taskset]["groups"][group2]["train"]["ruleset"]
                # if config[taskset]["groups"][group2]["train"]["frozen"] == False
                # else config[taskset]["groups"][group2]["pretrain"]["ruleset"]
            )
            if len(group2_tasks) == 0:
                shared_tasks = 0
            else:
                shared_tasks = 100 * len(group2_tasks) / 3
        elif group2 == "master":
            group1_tasks = (
                config[taskset]["groups"][group1]["pretrain"]["ruleset"]
                # + config[taskset]["groups"][group1]["train"]["ruleset"]
                # if config[taskset]["groups"][group1]["train"]["frozen"] == False
                # else config[taskset]["groups"][group1]["pretrain"]["ruleset"]
            )
            if len(group1_tasks) == 0:
                shared_tasks = 0
            else:
                shared_tasks = 100 * len(group1_tasks) / 3
        try:
            group_pairs[shared_tasks].append(pair)
        except KeyError:
            group_pairs[shared_tasks] = [pair]
    return group_pairs


def dissimilarities_per_percentage_of_shared_task(group_pairs, df):
    dissimilarities_per_shared_task = {
        "dsa": {perc: [] for perc in group_pairs.keys()},
        "cka": {perc: [] for perc in group_pairs.keys()},
        "procrustes": {perc: [] for perc in group_pairs.keys()},
    }
    for shared_tasks, pairs in group_pairs.items():
        for pair in pairs:
            group1, group2 = pair
            data_pair = df[
                ((df["group1"] == group1) & (df["group2"] == group2))
                # or the opposite
                | ((df["group1"] == group2) & (df["group2"] == group1))
            ]
            # if pair == ("pretrain_frozen", "pretrain_unfrozen"):
            #     print(data_pair)
            for measure in measures:
                data_pair_mesure = data_pair[data_pair["measure"] == measure][
                    "dissimilarity"
                ].tolist()
                dissimilarities_per_shared_task[measure][shared_tasks].extend(
                    data_pair_mesure
                )
    return dissimilarities_per_shared_task


def get_dissimilarities_groups(taskset):
    # take all the folder names under data/dissimilarities_over_learning/{taskset}
    groups_training = os.listdir(f"../data/dissimilarities_over_learning/{taskset}")
    groups_training = [group for group in groups_training if group != ".DS_Store"]
    dissimilarities_groups = {group: None for group in groups_training}

    for group_training in groups_training:
        path = f"../data/dissimilarities_over_learning/{taskset}/{group_training}"
        measures = ["cka", "dsa", "procrustes", "accuracy_1", "accuracy_2"]
        sampling = [0, 25, 50, 75, 100]
        dissimilarities = {measure: [] for measure in measures}

        for measure in measures:
            path_measure = os.path.join(path, measure)
            files = os.listdir(path_measure)
            for file in files:
                file_path = os.path.join(path_measure, file)
                if file_path.endswith(".npz"):
                    with np.load(file_path) as data:
                        dissimilarities[measure].append(data["arr_0"])
        dissimilarities_interpolated = {
            measure: {group: [] for group in range(len(sampling))}
            for measure in measures
        }
        for measure in measures:
            for dissimilarity in dissimilarities[measure]:
                if dissimilarity.shape[0] > 4:
                    dissimilarities_interpolated[measure][0].append(dissimilarity[0])
                    for i in range(len(sampling) - 1):
                        index_start = int(sampling[i] / 100 * (dissimilarity.shape[0]))
                        index_end = int(
                            sampling[i + 1] / 100 * (dissimilarity.shape[0])
                        )
                        dissimilarities_interpolated[measure][i + 1].append(
                            np.median(dissimilarity[index_start:index_end])
                        )
        for measure in measures:
            for group in range(len(sampling)):
                dissimilarities_interpolated[measure][group] = np.mean(
                    dissimilarities_interpolated[measure][group]
                )
        dissimilarities_groups[group_training] = dissimilarities_interpolated
    return dissimilarities_groups, groups_training


def get_dissimilarities_shared_task_shared_curriculum(
    group_pairs, dissimilarities_groups, x_values
):
    measures_selected = ["cka", "dsa", "procrustes", "accuracy_1", "accuracy_2"]
    diss_cc = {
        measure: {shared: [] for shared in group_pairs} for measure in measures_selected
    }
    for measure in diss_cc:
        for shared in diss_cc[measure]:
            for pair in group_pairs[shared]:
                name_1 = pair[0] + "_" + pair[1]
                name_2 = pair[1] + "_" + pair[0]
                if name_1 in dissimilarities_groups:
                    diss_cc[measure][shared].append(
                        [x_values, dissimilarities_groups[name_1][measure]]
                    )
                elif name_2 in dissimilarities_groups:
                    diss_cc[measure][shared].append(
                        [x_values, dissimilarities_groups[name_2][measure]]
                    )
            # once all the pairs are added, we can interpolate the values
            x_new = x_values
            y_new = []
            for i in range(len(x_values)):
                y_new.append(
                    np.nanmean([diss[1][i] for diss in diss_cc[measure][shared]])
                )
            diss_cc[measure][shared] = [x_new, y_new]
    return diss_cc
