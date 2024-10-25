import sys
import os
from turtle import st

from networkx import group_closeness_centrality

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from dsa_analysis import load_config, visualize
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
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

color_mapping = {
    "master": "#ED6A5A",  # Bittersweet (reddish-orange)
    "untrained": "#E5B25D",  # Hunyadi yellow (mustard-yellow)
    "master_frozen": "#9BC1BC",  # Ash gray (muted teal)
    "pretrain_partial": "#8FBC8F",  # Payne's gray (blue-gray)
    "pretrain_basic_frozen": "#8FBC8F",  # Payne's gray (blue-gray)
    "pretrain_anti_frozen": "#8FBC8F",  # Palatinate (deep purple)
    "pretrain_delay_frozen": "#8FBC8F",  # Darker shade of blue (medium blue)
    "pretrain_basic_anti_frozen": "#8FBC8F",  # Darker shade of blue (medium blue)
    "pretrain_basic_delay_frozen": "#8FBC8F",  # Darker shade of blue (medium blue
    "pretrain_frozen": "#4C1E4F",  # Palatinate (deep purple)
    "pretrain_unfrozen": "#696D7D",  # Palatinate (deep purple)
}

group_mapping_names = {
    "master": "Master",
    "untrained": "Untrained",
    "master_frozen": "Master & Frozen",
    "pretrain_partial": "Partial Pretraining",
    "pretrain_basic_frozen": "Pretrain Basic Frozen",
    "pretrain_frozen": "Full Pretraining",
    "pretrain_unfrozen": "Full Pretraining & Unfrozen",
    "pretrain_anti_frozen": "Pretrain Anti Frozen",
    "pretrain_delay_frozen": "Pretrain Delay Frozen",
    "pretrain_basic_anti_frozen": "Pretrain Basic Anti Frozen",
    "pretrain_basic_delay_frozen": "Pretrain Basic Delay Frozen",
}


color_mapping_metrics = {
    "dsa": "#7B1E3C",  # Nice green (medium green)
    "cka": "#8D99AE",  # Light shade of blue (sky blue)
    "procrustes": "#2B2F42",  # Darker shade of blue (medium blue)
}
# ["#8D99AE", "#2B2F42", "#EF233C"]  # Greys for CKA and Procrustes, Red for DSA


color_mapping_tasks = {
    "Delay": "#1F77B4",  # Medium blue
    "Delay Anti": "#FF7F0E",  # Bright orange
}

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


def get_dynamics_rnn(
    rnn_type, activation, hidden_size, lr, model, group, device, taskset
):
    # Load configuration and set hyperparameters
    config = load_config("../config.yaml")
    ruleset = config[taskset]["rules_analysis"][-1]
    all_rules = config[taskset]["rules_analysis"]

    hp = {
        "rnn_type": rnn_type,
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.00001,
        "l2_weight": 0.00001,
        "mode": "test",
    }
    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=ruleset
    )
    run_model = main.load_model(
        f"../models/{taskset}/{group}/{model}",
        hp,
        RNNLayer,
        device=device,
    )
    h = main.representation(run_model, all_rules, rnn=True)
    return h


def get_dynamics_rnn_task(
    rnn_type, activation, hidden_size, lr, model, group, device, task, taskset
):
    # Load configuration and set hyperparameters
    config = load_config("../config.yaml")
    ruleset = config[taskset]["rules_analysis"][-1]
    all_rules = config[taskset]["rules_analysis"]

    hp = {
        "rnn_type": rnn_type,
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.00001,
        "l2_weight": 0.00001,
        "mode": "test",
    }
    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=ruleset
    )
    run_model = main.load_model(
        f"../models/{taskset}/{group}/{model}",
        hp,
        RNNLayer,
        device=device,
    )
    h = main.representation_task(run_model, all_rules, task, rnn=True)
    return h


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
        "accuracy_1": [],
        "accuracy_2": [],
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
                data = np.load(file_path, allow_pickle=True)
                data_accuracy = np.load(
                    file_path.replace(measure, "accuracy"), allow_pickle=True
                )
                dissimilarities[measure].append(data["arr_0"])
                for i in range(len(groups)):
                    for j in range(len(groups)):
                        array_dissimilarities = remove_nan(
                            replace_nan_diagonal(data["arr_0"])
                        )
                        array_accuracy = remove_nan(data_accuracy["arr_0"])
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
                            df["accuracy_1"].append(array_accuracy[i][0])
                            df["accuracy_2"].append(array_accuracy[j][0])
    return pd.DataFrame(df)


def select_df(df):
    groups_trained = ["master"]
    # Condition for group1: if group1 is in critical_groups, then accuracy_1 should be 1
    condition1 = ~df["group1"].isin(groups_trained) | (df["accuracy_1"] == 1)

    # Condition for group2: if group2 is in critical_groups, then accuracy_2 should be 1
    condition2 = ~df["group2"].isin(groups_trained) | (df["accuracy_2"] == 1)

    # # Condition for activation to not be "leaky_relu"
    condition3 = df["activation"] != "leaky_relu"

    condition4 = df["activation"] != "softplus"

    condition5 = df["model_type"] != "leaky_rnn"

    # Filter DataFrame based on the combined conditions
    df_selected = df[condition1 & condition2 & condition3 & condition4 & condition5]
    # df_selected = df

    models_trained_per_group = {group + "_master": [] for group in groups_trained}
    for group in groups_trained:
        df_selected_group = df_selected[
            (df_selected["group1"] == group) | (df_selected["group2"] == group)
        ]
        for row, data in df_selected_group.iterrows():
            model = (
                data["model_type"]
                + "_"
                + data["activation"]
                + "_"
                + str(data["hidden_size"])
                + "_"
                + str(data["lr"])
                + "_"
                + str(data["batch_size"])
            )
            models_trained_per_group[group + "_master"].append(model)
    for group in models_trained_per_group:
        models_trained_per_group[group] = list(set(models_trained_per_group[group]))

    return df_selected, models_trained_per_group


# Define the mapping for group2
def map_group(group):
    if group in [
        "pretrain_basic_frozen",
        "pretrain_anti_frozen",
        "pretrain_delay_frozen",
        "pretrain_basic_anti_frozen",
        "pretrain_basic_delay_frozen",
    ]:
        return "pretrain_partial"
    return group


def t_standart_error_dissimilarity(df, group, measure):
    # get data
    data_group = df[
        (df["group1"] == group)
        & (df["group2"] == "master")
        & (df["measure"] == measure)  # or the opposite
        | (
            (df["group1"] == "master")
            & (df["group2"] == group)
            & (df["measure"] == measure)
        )
    ]["dissimilarity"]
    mean_dissimilarities = data_group.mean()
    n = len(data_group)
    standard_error = data_group.std() / np.sqrt(n)

    return mean_dissimilarities, standard_error


def t_test_dissimilarity(df, group1, group2, measure):
    # get data
    data_group1 = df[
        (df["group1"] == group1)
        & (df["group2"] == "master")
        & (df["measure"] == measure)  # or the opposite
        | (
            (df["group1"] == "master")
            & (df["group2"] == group1)
            & (df["measure"] == measure)
        )
    ]["dissimilarity"]
    data_group2 = df[
        (df["group1"] == group2)
        & (df["group2"] == "master")
        & (df["measure"] == measure)
        | (
            (df["group1"] == "master")
            & (df["group2"] == group2)
            & (df["measure"] == measure)
        )
    ]["dissimilarity"]
    # perform t-test
    t_stat, p_val = stats.ttest_ind(data_group1, data_group2)
    # stat, p_value = mannwhitneyu(data_group1, data_group2)
    return t_stat, p_val
    # return stat, p_value


# function to perform t-test on all pairs of groups for a given measure
def t_test_all_pairs(dg, measure):
    # map groups
    df = dg.copy()
    df["group2"] = df["group2"].apply(map_group)
    df["group1"] = df["group1"].apply(map_group)
    groups = [
        "untrained",
        "master_frozen",
        "pretrain_partial",
        "pretrain_frozen",
        "pretrain_unfrozen",
    ]
    p_values = []
    groups_pairs = []
    for group1 in groups:
        for group2 in groups:
            if group1 != group2:
                t_stat, p_val = t_test_dissimilarity(df, group1, group2, measure)
                p_values.append(p_val)
                groups_pairs.append((group1, group2))
    # adjust p-values
    adjusted_p_values = multipletests(p_values, method="fdr_bh")[1]
    # store results in a dataframe
    t_test_results_df = pd.DataFrame(
        {
            "pairs": groups_pairs,
            "p_value": p_values,
            "adjusted_p_value": adjusted_p_values,
        }
    )
    return t_test_results_df


# function to compute standard error intervals for all groups
def t_standart_error_dissimilarity_all_groups(dg, measure):
    # map groups
    df = dg.copy()
    df["group2"] = df["group2"].apply(map_group)
    df["group1"] = df["group1"].apply(map_group)
    groups = [
        "untrained",
        "master_frozen",
        "pretrain_partial",
        "pretrain_frozen",
        "pretrain_unfrozen",
    ]
    mean_dissimilarities = []
    standard_errors = []
    for group in groups:
        mean_diss, std_error = t_standart_error_dissimilarity(df, group, measure)
        mean_dissimilarities.append(mean_diss)
        standard_errors.append(std_error)
    # store results in a dataframe
    t_test_results_df = pd.DataFrame(
        {
            "group": groups,
            "mean_dissimilarities": mean_dissimilarities,
            "standard_errors": standard_errors,
        }
    )
    return t_test_results_df


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
                / len(set(group1_tasks).union(set(group2_tasks)))
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
        pair for pair in pairs if pair[0] != "untrained" and pair[1] != "untrained"
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
        "dsa": {perc: {} for perc in group_pairs.keys()},
        "cka": {perc: {} for perc in group_pairs.keys()},
        "procrustes": {perc: {} for perc in group_pairs.keys()},
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
            for measure in measures:
                data_pair_measure = data_pair[data_pair["measure"] == measure]
                for i, row in data_pair_measure.iterrows():
                    model = (
                        row["model_type"]
                        + "_"
                        + row["activation"]
                        + "_"
                        + str(row["hidden_size"])
                        + "_"
                        + str(row["lr"])
                        + "_"
                        + str(row["batch_size"])
                    )
                    dissimilarities_per_shared_task[measure][shared_tasks][model] = []
                    dissimilarities_per_shared_task[measure][shared_tasks][
                        model
                    ].append(row["dissimilarity"])
    return dissimilarities_per_shared_task


def get_dissimilarities_groups(taskset, models_trained_per_group):
    # take all the folder names under data/dissimilarities_over_learning/{taskset}
    groups_training = os.listdir(f"../data/dissimilarities_over_learning/{taskset}")
    groups_training = [
        group for group in groups_training if group != ".DS_Store" and "master" in group
    ]
    # groups_training = [group for group in groups_training if "master" in group]
    dissimilarities_groups = {group: None for group in groups_training}

    for group_training in groups_training:
        path = f"../data/dissimilarities_over_learning/{taskset}/{group_training}"
        measures = ["cka", "dsa", "procrustes", "accuracy_1", "accuracy_2"]
        sampling = [i * 20 for i in range(6)]
        dissimilarities = {measure: [] for measure in measures}

        for measure in measures:
            path_measure = os.path.join(path, measure)
            files = os.listdir(path_measure)
            for file in files:
                model_name = file.replace(".npz", "")
                if group_training in models_trained_per_group:
                    if model_name in models_trained_per_group[group_training]:
                        file_path = os.path.join(path_measure, file)
                        if file_path.endswith(".npz"):
                            with np.load(file_path) as data:
                                dissimilarities[measure].append(data["arr_0"])
                else:
                    file_path = os.path.join(path_measure, file)
                    if file_path.endswith(".npz"):
                        with np.load(file_path) as data:
                            dissimilarities[measure].append(data["arr_0"])
        dissimilarities_interpolated = {
            measure: {group: [] for group in range(len(sampling))}
            for measure in measures
        }

        # average to have the sampling
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

        # average over the groups
        for measure in measures:
            for group in range(len(sampling)):
                dissimilarities_interpolated[measure][group] = (
                    np.nanmean(dissimilarities_interpolated[measure][group]),
                    np.nanstd(dissimilarities_interpolated[measure][group])
                    / np.sqrt(len(dissimilarities_interpolated[measure][group])),
                    dissimilarities_interpolated[measure][group],
                )
        dissimilarities_groups[group_training] = dissimilarities_interpolated
    return dissimilarities_groups, groups_training


def get_dissimilarities_shared_task_shared_curriculum(
    group_pairs, dissimilarities_groups, x_values
):
    measures_selected = ["cka", "dsa", "procrustes"]
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
                        [
                            x_values,
                            [
                                dissimilarities_groups[name_1][measure][x][2]
                                for x in range(len(x_values))
                            ],
                        ]
                    )
                elif name_2 in dissimilarities_groups:
                    diss_cc[measure][shared].append(
                        [
                            x_values,
                            [
                                dissimilarities_groups[name_2][measure][x][2]
                                for x in range(len(x_values))
                            ],
                        ]
                    )
            # once all the pairs are added, we can interpolate the values
            x_new = x_values
            y_new = []
            y_new_std = []
            for i in range(len(x_values)):
                y_new.append(
                    np.nanmean(
                        [
                            diss_item
                            for diss in diss_cc[measure][shared]
                            for diss_item in diss[1][i]
                        ]
                    )
                )
                y_new_std.append(
                    np.nanstd(
                        [
                            diss_item
                            for diss in diss_cc[measure][shared]
                            for diss_item in diss[1][i]
                        ]
                    )
                    / np.sqrt(
                        len(
                            [
                                diss_item
                                for diss in diss_cc[measure][shared]
                                for diss_item in diss[1][i]
                            ]
                        )
                    )
                )
            diss_cc[measure][shared] = [x_new, y_new, y_new_std]
    return diss_cc
