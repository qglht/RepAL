import sys
import os

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
from mambapy.mamba_lm import MambaLM, MambaLMConfig
from mambapy.mamba import Mamba, MambaConfig, RMSNorm
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

color_mapping = {
    "master": "#4D4D4D",  # Soft shade of black (charcoal gray)
    "untrained": "#E57373",  # Soft shade of red (light red)
    "master_frozen": "#FFB74D",  # Soft shade of orange (light orange)
    "pretrain_partial": "#81C784",  # Soft shade of light green (pale green)
    "pretrain_basic_frozen": "#81C784",  # Muted green (dark green)
    "pretrain_frozen": "#2E7D32",  # Shade of dark green (forest green)
    "pretrain_unfrozen": "#1565C0",  # Shade of dark blue (navy blue)
}

color_mapping_metrics = {
    "dsa": "#66BB6A",  # Nice green (medium green)
    "cka": "#42A5F5",  # Light shade of blue (sky blue)
    "procrustes": "#1E88E5",  # Darker shade of blue (medium blue)
}


color_mapping_tasks = {
    "Delay": "#1F77B4",  # Medium blue
    "Delay Anti": "#FF7F0E",  # Bright orange
}

measures = ["cka", "dsa", "procrustes"]


def parse_model_info(model_name):
    model_name = model_name.replace(".pth", "")
    model_name = model_name.split("_")
    d_model = int(model_name[1])
    n_layers = int(model_name[2])
    learning_rate = float(model_name[3])
    batch_size = int(model_name[4])
    return d_model, n_layers, learning_rate, batch_size


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


def get_dynamics_mamba(
    d_model,
    n_layers,
    learning_rate,
    batch_size,
    model,
    group,
    taskset,
    device,
):
    # Load configuration and set hyperparameters
    config = load_config("../config.yaml")
    ruleset = config[taskset]["rules_analysis"][-1]
    all_rules = config[taskset]["rules_analysis"]

    hp = {
        "num_epochs": 50,
        "batch_size_train": batch_size,
        "learning_rate": learning_rate,
        "l2_weight": 0.0001,
        "mode": "test",
    }
    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=ruleset
    )
    config = MambaLMConfig(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=hp["n_input"],
        pad_vocab_size_multiple=1,  # https://github.com/alxndrTL/mamba.py/blob/main/mamba_lm.py#L27
        pscan=True,
    )
    run_model = main.load_model_mamba(
        f"../models/mamba/{taskset}/{group}/{model}",
        hp,
        config,
        device=device,
    )
    h = main.representation(run_model, all_rules, rnn=False)
    return h


def get_dataframe(path, taskset):
    path = f"../data/dissimilarities/mamba/{taskset}/"
    df = {
        "d_model": [],
        "n_layers": [],
        "learning_rate": [],
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
                d_model, n_layers, learning_rate, batch_size = parse_model_info(file)
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
                            df["d_model"].append(d_model)
                            df["n_layers"].append(n_layers)
                            df["learning_rate"].append(learning_rate)
                            df["batch_size"].append(batch_size)
                            df["group1"].append(groups[i])
                            df["group2"].append(groups[j])
                            df["measure"].append(measure)
                            df["dissimilarity"].append(array_dissimilarities[i, j])
                            df["accuracy_1"].append(array_accuracy[i][0])
                            df["accuracy_2"].append(array_accuracy[j][0])
    return pd.DataFrame(df)


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


# function to compute standard error intervals for all groups
def t_standart_error_dissimilarity_all_groups(dg, measure):
    # map groups
    df = dg.copy()
    df["group2"] = df["group2"].apply(map_group)
    df["group1"] = df["group1"].apply(map_group)
    print(df["group2"].unique())
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


# function to perform t-test on all pairs of groups for a given measure
def t_test_all_pairs(dg, measure):
    # map groups
    df = dg.copy()
    df["group2"] = df["group2"].apply(map_group)
    df["group1"] = df["group1"].apply(map_group)
    print(df["group2"].unique())
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
    groups_training = os.listdir(
        f"../data/dissimilarities_over_learning/mamba/{taskset}"
    )
    groups_training = [group for group in groups_training if group != ".DS_Store"]
    dissimilarities_groups = {group: None for group in groups_training}

    for group_training in groups_training:
        path = f"../data/dissimilarities_over_learning/mamba/{taskset}/{group_training}"
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
                if dissimilarity.shape[0] > 2:
                    dissimilarities_interpolated[measure][0].append(dissimilarity[0])
                    for i in range(len(sampling) - 1):
                        index_start = int(sampling[i] / 100 * (dissimilarity.shape[0]))
                        index_end = int(
                            sampling[i + 1] / 100 * (dissimilarity.shape[0])
                        )
                        dissimilarities_interpolated[measure][i + 1].append(
                            np.nanmedian(dissimilarity[index_start:index_end])
                        )
        for measure in measures:
            for group in range(len(sampling)):
                dissimilarities_interpolated[measure][group] = np.nanmedian(
                    dissimilarities_interpolated[measure][group]
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
                    np.nanmedian([diss[1][i] for diss in diss_cc[measure][shared]])
                )
            diss_cc[measure][shared] = [x_new, y_new]
    return diss_cc
