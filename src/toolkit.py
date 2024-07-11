from pdb import run
import warnings

from matplotlib.pylab import f

import main
from main import RNNLayer
from dsa_analysis import load_config
import torch
import os
import pandas as pd
from dsa_analysis import (
    simulation_line,
    simulation_lorenz,
    combine_simulations,
    load_config,
)
import DSA
import pandas as pd
import numpy as np
from itertools import permutations
from main.train import accuracy
import similarity
import copy
from collections import OrderedDict
import torch
from torch import nn, jit
from mambapy.mamba_lm import MambaLM, MambaLMConfig
from mambapy.mamba import Mamba, MambaConfig, RMSNorm

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def same_order(comp_motif_1, comp_motif_2) -> bool:
    return len(
        [i for i in range(len(comp_motif_1)) if comp_motif_1[i] == comp_motif_2[i]]
    )


def find_checkpoints(name):
    # Find the latest checkpoint file
    checkpoint_dir = name
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("epoch_") and f.endswith("_checkpoint.pth")
    ]
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1]))
    return checkpoint_files


def initialize_model(
    taskset, rnn_type, activation, hidden_size, lr, batch_size, device
):
    config = load_config("config.yaml")
    all_rules = config[taskset]["rules_analysis"]
    hp = {
        "rnn_type": rnn_type,
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.0001,
        "l2_weight": 0.0001,
        "num_epochs": 50,
        "batch_size_train": batch_size,
        "mode": "test",
    }
    hp, log, optimizer = main.set_hyperparameters(
        model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=all_rules
    )
    run_model = main.Run_Model(hp, RNNLayer, device)

    return run_model, hp


def load_model_jit(run_model_copy, checkpoint):
    # Load state dict into the new model, handling scripting:
    if isinstance(run_model_copy, torch.jit.RecursiveScriptModule):
        # Remove "_c." prefix from keys for scripted models
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith("_c."):
                name = k[3:]  # remove `_c.` prefix
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        run_model_copy.load_state_dict(new_state_dict)
    else:
        # Load state dict directly for non-scripted models
        run_model_copy.load_state_dict(checkpoint["model_state_dict"])

    return run_model_copy


def corresponding_training_time(n, p):
    # Find the argmin over j for each i
    return [
        min(range(p), key=lambda j: abs(int(100 * i / n) - int(100 * j / p)))
        for i in range(n)
    ]


def get_curves(taskset, model, rules, components):
    h = main.representation(model, rules)
    h_trans, _ = main.compute_pca(h, n_components=components)
    if taskset == "PDM":
        tensor_on_cpu = h_trans[
            ("AntiPerceptualDecisionMakingDelayResponseT", "stimulus")
        ].cpu()
    else:
        tensor_on_cpu = h_trans[("AntiGoNogoDelayResponseT", "stimulus")].cpu()
    return tensor_on_cpu.detach().numpy()


def normalize_within_unit_volume(tensor):
    # Ensure the input is a PyTorch tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")

    # Find the minimum and maximum values in the entire 3D tensor
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)

    # Calculate scaling factor to fit the entire tensor within the unit volume
    scale_factor = 1.0 / (max_value - min_value)

    # Normalize the tensor
    normalized_tensor = (tensor - min_value) * scale_factor
    # convert tensor to numpy array

    return normalized_tensor


def pipeline(taskset, group, rnn_type, activation, hidden_size, lr, batch_size, device):
    config = load_config("config.yaml")
    rules_pretrain = config[taskset]["groups"][group]["pretrain"]["ruleset"]
    rules_train = config[taskset]["groups"][group]["train"]["ruleset"]
    freeze = config[taskset]["groups"][group]["train"]["frozen"]
    all_rules = config[taskset]["all_rules"]
    hp = {
        "rnn_type": rnn_type,
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.0001,
        "l2_weight": 0.0001,
        "num_epochs": 50,
        "batch_size_train": batch_size,
    }
    model_name = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}"
    path_pretrain_folder = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_pretrain"
    )
    path_pretrain_model = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_pretrain.pth"
    )
    path_train_folder = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_train"
    )
    path_train_model = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_train.pth"
    )

    # Pretraining
    print(f"Pretraining model {model_name} for group {group} and taskset {taskset}")
    if not os.path.exists(path_pretrain_model):
        if rules_pretrain:
            hp, log, optimizer = main.set_hyperparameters(
                model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_pretrain
            )
            run_model = main.Run_Model(hp, RNNLayer, device)
            main.train(run_model, optimizer, hp, log, path_pretrain_folder)
            run_model.save(path_pretrain_model)

    # Training
    print(f"Training model {model_name} for group {group}")
    if not os.path.exists(path_train_model):
        if rules_train:
            if rules_pretrain:
                # load the model first
                hp, log, optimizer = main.set_hyperparameters(
                    model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_train
                )
                run_model = main.load_model(
                    path_pretrain_model,
                    hp,
                    RNNLayer,
                    device=device,
                )
                main.train(
                    run_model, optimizer, hp, log, path_train_folder, freeze=freeze
                )
                run_model.save(path_train_model)
            else:
                hp, log, optimizer = main.set_hyperparameters(
                    model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_train
                )
                run_model = main.Run_Model(hp, RNNLayer, device)
                main.train(
                    run_model, optimizer, hp, log, path_train_folder, freeze=freeze
                )
                run_model.save(path_train_model)
        else:
            # if rules_train is empty, then we don't train the model
            hp, log, optimizer = main.set_hyperparameters(
                model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_train
            )
            run_model = main.Run_Model(hp, RNNLayer, device)
            run_model.save(path_train_model)
    return


def pipeline_mamba(
    taskset,
    group,
    d_model,
    n_layers,
    pad_vocab_size_multiple,
    pscan,
    learning_rate,
    batch_size,
    device,
):

    config = load_config("config.yaml")
    rules_pretrain = config[taskset]["groups"][group]["pretrain"]["ruleset"]
    rules_train = config[taskset]["groups"][group]["train"]["ruleset"]
    freeze = config[taskset]["groups"][group]["train"]["frozen"]
    all_rules = config[taskset]["all_rules"]
    hp = {
        "num_epochs": 50,
        "batch_size_train": batch_size,
        "learning_rate": learning_rate,
    }
    model_name = f"mamba_{d_model}_{n_layers}_{learning_rate}_{batch_size}"
    path_pretrain_folder = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_pretrain"
    )
    path_pretrain_model = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_pretrain.pth"
    )
    path_train_folder = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_train"
    )
    path_train_model = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_train.pth"
    )

    # Pretraining
    print(f"Pretraining model {model_name} for group {group} and taskset {taskset}")
    if not os.path.exists(path_pretrain_model):
        if rules_pretrain:
            hp, log, optimizer = main.set_hyperparameters(
                model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_pretrain
            )
            config = MambaLMConfig(
                d_model=d_model,
                n_layers=n_layers,
                vocab_size=hp["n_input"],
                pad_vocab_size_multiple=pad_vocab_size_multiple,  # https://github.com/alxndrTL/mamba.py/blob/main/mamba_lm.py#L27
                pscan=pscan,
            )
            run_model = main.MambaSupervGym(
                hp["n_output"], hp["n_input"], config, device=device
            )
            main.train(run_model, optimizer, hp, log, path_pretrain_folder)
            run_model.save(path_pretrain_model)

    return


def generate_data(taskset, env):
    config = load_config("config.yaml")
    all_rules = config[taskset]["all_rules"]
    hp, _, _ = main.set_hyperparameters(model_dir="debug", hp={}, ruleset=all_rules)
    main.generate_data(env, hp, mode="train", num_pregenerated=10000)

    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp={"mode": "test"}, ruleset=all_rules
    )
    main.generate_data(env, hp, mode="test", num_pregenerated=1000)


def get_dynamics_model(
    rnn_type, activation, hidden_size, lr, model, group, taskset, device, n_components=3
):
    # Load configuration and set hyperparameters
    config = load_config("config.yaml")
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
        f"models/{taskset}/{group}/{model}",
        hp,
        RNNLayer,
        device=device,
    )
    h = main.representation(run_model, all_rules)
    h_trans, explained_variance = main.compute_pca(h, n_components=n_components)
    if taskset == "PDM":
        tensor_on_cpu = h_trans[
            ("AntiPerceptualDecisionMakingDelayResponseT", "stimulus")
        ].cpu()
    else:
        tensor_on_cpu = h_trans[("AntiGoNogoDelayResponseT", "stimulus")].cpu()
    return tensor_on_cpu.detach().numpy(), explained_variance


def dissimilarity_over_learning(
    taskset, group1, group2, rnn_type, activation, hidden_size, lr, batch_size, device
):
    config = load_config("config.yaml")
    all_rules = config[taskset]["rules_analysis"]

    # paths for checkpoints
    model_name = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}"
    path_train_folder1 = os.path.join(
        f"models/{taskset}/{group1}", model_name + f"_train"
    )
    path_train_folder2 = os.path.join(
        f"models/{taskset}/{group2}", model_name + f"_train"
    )

    # initialize model architectures
    run_model1, hp1 = initialize_model(
        taskset, rnn_type, activation, hidden_size, lr, batch_size, device
    )
    run_model2, hp2 = initialize_model(
        taskset, rnn_type, activation, hidden_size, lr, batch_size, device
    )

    # get checkpoints in train path
    checkpoint_files_1 = find_checkpoints(path_train_folder1)
    checkpoint_files_2 = find_checkpoints(path_train_folder2)

    # group models and establish correspondancy between epochs
    models_to_compare = []

    # if pretrain in group1 and group2, load checkpoints at os.path.join(f"models/{group}", model_name + f"_pretrain.pth")
    if "pretrain" in group1 and "pretrain" in group2:
        path_pretrain_1 = os.path.join(
            f"models/{taskset}/{group1}", model_name + f"_pretrain.pth"
        )
        path_pretrain_2 = os.path.join(
            f"models/{taskset}/{group2}", model_name + f"_pretrain.pth"
        )
        run_model1_pretrain = main.load_model(
            path_pretrain_1,
            hp1,
            RNNLayer,
            device=device,
        )
        run_model2_pretrain = main.load_model(
            path_pretrain_2,
            hp2,
            RNNLayer,
            device=device,
        )
        models_to_compare.extend([(run_model1_pretrain, run_model2_pretrain)])

    dissimilarities_over_learning = {
        "cka": [],
        "dsa": [],
        "procrustes": [],
        "accuracy_1": [],
        "accuracy_2": [],
    }
    cka_measure = similarity.make("measure.sim_metric.cka-angular-score")
    procrustes_measure = similarity.make("measure.netrep.procrustes-angular-score")
    if checkpoint_files_1 and checkpoint_files_2:
        # get the models to compare
        if len(checkpoint_files_1) < len(checkpoint_files_2):
            index_epochs = corresponding_training_time(
                len(checkpoint_files_1), len(checkpoint_files_2)
            )
            for epoch in index_epochs:
                run_model1_copy = copy.deepcopy(run_model1)
                run_model2_copy = copy.deepcopy(run_model2)
                checkpoint1 = torch.load(
                    os.path.join(
                        path_train_folder1,
                        checkpoint_files_1[index_epochs.index(epoch)],
                    ),
                    map_location=device,
                )
                run_model1_copy = load_model_jit(run_model1_copy, checkpoint1)
                accuracy_1 = float(checkpoint1["log"]["perf_min"][-1])
                checkpoint2 = torch.load(
                    os.path.join(path_train_folder2, checkpoint_files_2[epoch]),
                    map_location=device,
                )
                run_model2_copy = load_model_jit(run_model2_copy, checkpoint2)
                accuracy_2 = float(checkpoint2["log"]["perf_min"][-1])
                models_to_compare.extend([(run_model1_copy, run_model2_copy)])
                dissimilarities_over_learning["accuracy_1"].append(accuracy_1)
                dissimilarities_over_learning["accuracy_2"].append(accuracy_2)
        else:
            index_epochs = corresponding_training_time(
                len(checkpoint_files_2), len(checkpoint_files_1)
            )
            for epoch in index_epochs:
                run_model1_copy = copy.deepcopy(run_model1)
                run_model2_copy = copy.deepcopy(run_model2)
                checkpoint1 = torch.load(
                    os.path.join(path_train_folder1, checkpoint_files_1[epoch]),
                    map_location=device,
                )
                run_model1_copy = load_model_jit(run_model1_copy, checkpoint1)
                accuracy_1 = float(checkpoint1["log"]["perf_min"][-1])
                checkpoint2 = torch.load(
                    os.path.join(
                        path_train_folder2,
                        checkpoint_files_2[index_epochs.index(epoch)],
                    ),
                    map_location=device,
                )
                run_model2_copy = load_model_jit(run_model2_copy, checkpoint2)
                accuracy_2 = float(checkpoint2["log"]["perf_min"][-1])
                models_to_compare.extend([(run_model1_copy, run_model2_copy)])
                dissimilarities_over_learning["accuracy_1"].append(accuracy_1)
                dissimilarities_over_learning["accuracy_2"].append(accuracy_2)

        # compute the curves for models and dissimilarities
        curves = [
            (
                get_curves(taskset, tuple_model[0], all_rules, components=15),
                get_curves(taskset, tuple_model[1], all_rules, components=15),
            )
            for tuple_model in models_to_compare
        ]
        for epoch_index in range(len(index_epochs)):
            dissimilarities_over_learning["cka"].append(
                1 - cka_measure(curves[epoch_index][0], curves[epoch_index][1])
            )
            dissimilarities_over_learning["procrustes"].append(
                1 - procrustes_measure(curves[epoch_index][0], curves[epoch_index][1])
            )
            dsa_comp = DSA.DSA(
                curves[epoch_index][0],
                curves[epoch_index][1],
                n_delays=config["dsa"]["n_delays"],
                rank=config["dsa"]["rank"],
                delay_interval=config["dsa"]["delay_interval"],
                verbose=True,
                iters=1000,
                lr=1e-2,
                device=device,
            )
            dissimilarities_over_learning["dsa"].append(dsa_comp.fit_score())

    for key, value in dissimilarities_over_learning.items():
        dissimilarities_over_learning[key] = np.array(value)

    return dissimilarities_over_learning


def dissimilarity_within_learning(
    taskset, group, rnn_type, activation, hidden_size, lr, batch_size, device
):
    config = load_config("config.yaml")
    all_rules = config[taskset]["rules_analysis"]
    sampling = [0, 25, 50, 75, 100]

    # paths for checkpoints
    model_name = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}"
    path_train_folder = os.path.join(
        f"models/{taskset}/{group}", model_name + f"_train"
    )

    # initialize model architectures
    run_model, hp = initialize_model(
        taskset, rnn_type, activation, hidden_size, lr, batch_size, device
    )

    # get checkpoints in train path
    checkpoint_files = find_checkpoints(path_train_folder)

    # group models and establish correspondancy between epochs
    models_to_compare = []

    # if pretrain in group1 and group2, load checkpoints at os.path.join(f"models/{group}", model_name + f"_pretrain.pth")
    if "pretrain" in group:
        path_pretrain = os.path.join(
            f"models/{taskset}/{group}", model_name + f"_pretrain.pth"
        )
        run_model_pretrain = main.load_model(
            path_pretrain,
            hp,
            RNNLayer,
            device=device,
        )
        models_to_compare.extend([run_model_pretrain])

    cka_similarities = np.empty((len(sampling) - 1, len(sampling) - 1))
    procrustes_similarities = np.empty((len(sampling) - 1, len(sampling) - 1))
    dsa_similarities = np.empty((len(sampling) - 1, len(sampling) - 1))
    accuracies = []
    accuracies_grouped = []

    cka_measure = similarity.make("measure.sim_metric.cka-angular-score")
    procrustes_measure = similarity.make("measure.netrep.procrustes-angular-score")

    if checkpoint_files:
        if len(checkpoint_files) > 3:
            # load all the checkpoints
            for epoch in range(len(checkpoint_files)):
                checkpoint = torch.load(
                    os.path.join(path_train_folder, checkpoint_files[epoch]),
                    map_location=device,
                )
                run_model_copy = copy.deepcopy(run_model)
                run_model_copy = load_model_jit(run_model_copy, checkpoint)

                accuracy = float(checkpoint["log"]["perf_min"][-1])
                # convert accuracy to float if it was a string
                models_to_compare.extend([run_model_copy])
                accuracies.append(accuracy)

            print(f"computing representations for model {model_name} for group {group}")
            print(f"len checkpoints : {len(checkpoint_files)}")
            # compute the curves for models and dissimilarities

            curves = []
            for model in models_to_compare:
                curves.append(get_curves(taskset, model, all_rules, components=15))
            print(f"grouping accuracies for model {model_name} for group {group}")

            groups = []
            for i in range(len(sampling) - 1):
                index_start = int(sampling[i] * len(curves) / 100)
                index_end = int((sampling[i + 1]) * len(curves) / 100)
                groups.append(curves[index_start:index_end])
                accuracies_grouped.append(np.mean(accuracies[index_start:index_end]))

            # compute similarities across groups gathered by sampling
            print(f"computing similarities for model {model_name} for group {group}")
            print(f"Len groups : {len(groups)}")

            group_done = 0
            for i in range(len(groups)):
                for j in range(i, len(groups)):
                    print(f"perc group done : {100*group_done/(len(groups)**2)}")
                    group_done += 1
                    dissimilarities_cka = []
                    dissimilarities_procrustes = []
                    dissimilarities_dsa = []
                    p = min(len(groups[i]), len(groups[j]))
                    if p > 0:
                        for index_model1 in range(p):
                            for index_model2 in range(index_model1, p):
                                print(f"index model 1 : {100*index_model1/p}")
                                model1 = groups[i][index_model1]
                                model2 = groups[j][index_model2]

                                dissimilarities_cka.append(
                                    1 - cka_measure(model1, model2)
                                )
                                dissimilarities_procrustes.append(
                                    1 - procrustes_measure(model1, model2)
                                )
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
                                dissimilarities_dsa.append(dsa_comp.fit_score())

                        cka_similarities[i, j] = np.mean(dissimilarities_cka)
                        cka_similarities[j, i] = cka_similarities[i, j]
                        procrustes_similarities[i, j] = np.mean(
                            dissimilarities_procrustes
                        )
                        procrustes_similarities[j, i] = procrustes_similarities[i, j]
                        dsa_similarities[i, j] = np.mean(dissimilarities_dsa)
                        dsa_similarities[j, i] = dsa_similarities[i, j]

            print(f"similarities finished for model {model_name} for group {group}")

    accuracies_grouped = np.array(accuracies_grouped)
    return {
        "cka": cka_similarities,
        "procrustes": procrustes_similarities,
        "dsa": dsa_similarities,
        "accuracy": accuracies_grouped,
    }


def dsa_optimisation_compositionality(rank, n_delays, delay_interval, device, ordered):
    path_file = (
        f"data/dsa_results/{rank}_{n_delays}_{delay_interval}.csv"
        if ordered == False
        else f"data/dsa_results/{rank}_{n_delays}_{delay_interval}_ordered.csv"
    )
    print(f"Saving to: {path_file}")
    config = load_config("config.yaml")
    # Define parameters
    dt = config["simulations"]["dt"]
    num_steps = config["simulations"]["num_steps"]
    num_samples = config["simulations"]["num_samples"]
    lorenz_parameters = config["simulations"]["lorenz_parameters"]

    # Run simulations line
    simulations_line = simulation_line(num_steps, num_samples)

    # Run simulations curve
    simulations_curve = simulation_lorenz(
        dt, lorenz_parameters["one_attractor"][1], num_samples, num_steps
    )

    # Run simulations Pattern1
    simulations_pattern1 = simulation_lorenz(
        dt, lorenz_parameters["two_stable_attractors"][0], num_samples, num_steps
    )

    # Run simulations Pattern2
    simulations_pattern2 = simulation_lorenz(
        dt, lorenz_parameters["two_stable_attractors"][2], num_samples, num_steps
    )

    # Run simulations line-curve-line-curve
    combined_simulations_line_curve_line = combine_simulations(
        [
            simulations_line,
            simulations_curve,
            np.flip(simulations_line, axis=0),
            np.flip(simulations_curve, axis=0),
        ],
        method="attach",
    )

    motif_basis = [
        simulations_line,
        simulations_curve,
        simulations_pattern1,
        simulations_pattern2,
        combined_simulations_line_curve_line,
    ]
    motif_names = ["Line", "Curve", "Pattern1", "Pattern2", "Line-Curve-Line-Curve"]
    motif_dict = {motif_names[i]: motif_basis[i] for i in range(len(motif_basis))}
    all_simulations_length_3 = list(permutations(motif_names, 3))
    all_simulations_combined = {
        permutation: combine_simulations(
            [
                motif_dict[permutation[0]],
                motif_dict[permutation[1]],
                motif_dict[permutation[2]],
            ],
            method="attach",
        )
        for permutation in all_simulations_length_3
    }

    model = list(all_simulations_combined.values())
    model_names = list(all_simulations_combined.keys())

    dsa = DSA.DSA(
        model,
        n_delays=n_delays,
        rank=rank,
        delay_interval=delay_interval,
        verbose=True,
        iters=1000,
        lr=1e-2,
        device=device,
    )
    similarities = dsa.fit_score()

    grouped_by_shared_elements = {i: [] for i in range(4)}
    for comp_motif_1 in model_names:
        for comp_motif_2 in model_names:
            if ordered:
                grouped_by_shared_elements[
                    same_order(comp_motif_1, comp_motif_2)
                ].extend([(comp_motif_1, comp_motif_2)])
            else:
                set_1 = set(comp_motif_1)
                set_2 = set(comp_motif_2)
                grouped_by_shared_elements[len(set_1.intersection(set_2))].extend(
                    [(comp_motif_1, comp_motif_2)]
                )

    similarities_grouped_by_shared_elements = {i: [] for i in range(4)}
    for key in grouped_by_shared_elements:
        for tuple1, tuple2 in grouped_by_shared_elements[key]:
            similarities_grouped_by_shared_elements[key].append(
                similarities[model_names.index(tuple1), model_names.index(tuple2)]
            )

    # # compute median of similarities for each group and plot similarity vs number of shared elements
    # median_similarities = {key: np.median(value) for key, value in similarities_grouped_by_shared_elements.items()}
    # std_devs = {key: np.std(value) for key, value in similarities_grouped_by_shared_elements.items()}

    # # Prepare data for plotting
    # keys = list(median_similarities.keys())
    # median_values = list(median_similarities.values())
    # std_dev_values = list(std_devs.values())

    # df = pd.DataFrame({'Number of shared elements': keys, 'Median similarity': median_values, 'Standard deviation': std_dev_values})
    # Prepare lists to store DataFrame rows
    data = []

    # Iterate over the shared elements
    for (
        num_shared_elements,
        similarities,
    ) in similarities_grouped_by_shared_elements.items():
        tuples = grouped_by_shared_elements[num_shared_elements]

        # Zip the similarities with the corresponding element pairs
        for (element1, element2), similarity in zip(tuples, similarities):
            # Sort the elements to ensure uniqueness
            sorted_pair = sorted([element1, element2])
            data.append(
                [num_shared_elements, sorted_pair[0], sorted_pair[1], similarity]
            )

    # Create DataFrame
    df = pd.DataFrame(
        data,
        columns=["number of shared elements", "element1", "element2", "similarity"],
    )

    # Drop duplicates
    df = df.drop_duplicates(subset=["element1", "element2"])

    # check if the directory exists
    if not os.path.exists("data/dsa_results"):
        os.makedirs("data/dsa_results")
    print(f"saving file to {path_file}")
    df.to_csv(path_file)
    return
