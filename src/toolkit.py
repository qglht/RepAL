import warnings

import main
from main import RNNLayer
from dsa_analysis import load_config
import torch
import os
import pandas as pd
from dsa_analysis import simulation_line, simulation_lorenz, combine_simulations, load_config
import DSA
import pandas as pd
import numpy as np
from itertools import permutations
import ipdb

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'

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


def train_model(activation, hidden_size, lr, freeze, mode, no_pretraining, device):
    # Load configuration and set hyperparameters
    config = load_config("config.yaml")
    ruleset = config["rnn"][mode]["ruleset"]
    all_rules = config["rnn"]["train"]["ruleset"] + config["rnn"]["pretrain"]["ruleset"]
    num_epochs = config["rnn"][mode]["num_epochs"]
    hp = {
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.000001,
        "l2_weight": 0.000001,
        "num_epochs": num_epochs,
    }
    hp, log, optimizer = main.set_hyperparameters(
        model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=ruleset
    )

    model_name = f"{activation}_{hidden_size}_{lr}"
    if mode == "train":
        if no_pretraining:
            name = os.path.join("models", model_name + f"__{freeze}_train_nopretrain")
            if  model_name + f"__{freeze}_train_nopretrain.pth" in os.listdir("models"):
                return
            else:
                run_model = main.Run_Model(hp, RNNLayer, device)
                main.train(run_model, optimizer, hp, log, name, freeze=freeze)
                run_model.save(name+".pth")
        else: 
            name = os.path.join("models", model_name + f"__{freeze}_train_pretrain") 
            if  model_name + f"__{freeze}_train_pretrain.pth" in os.listdir("models"):
                return
            else:
                run_model = main.load_model(
                    f"models/{activation}_{hidden_size}_{lr}_pretrain.pth",
                    hp,
                    RNNLayer,
                    device=device,
                )
                main.train(run_model, optimizer, hp, log, name, freeze=freeze)
                run_model.save(name+".pth")
    elif mode == "pretrain":
        name = os.path.join("models", model_name + f"_pretrain")
        if model_name + f"_pretrain.pth" in os.listdir("models"):
            return
        else:
            run_model = main.Run_Model(hp, RNNLayer, device)
            main.train(run_model, optimizer, hp, log, name)
            run_model.save(name+".pth")
    return run_model

def generate_data(env):
    config = load_config("config.yaml")
    all_rules = config["rnn"]["rules"]
    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp={}, ruleset=all_rules
    )
    main.generate_data(env, hp, mode="train", num_pregenerated=10000)

    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp={"mode": "test"}, ruleset=all_rules
    )
    main.generate_data(env, hp, mode="test", num_pregenerated=1000)

def task_relevant_variables():
    return NotImplementedError


def compute_dissimilarity(activation, hidden_size, lr, freeze, nopretrain ,device, n_components=3):
    # Load configuration and set hyperparameters
    config = load_config("../config.yaml")
    ruleset = config["rnn"]["train"]["ruleset"]
    all_rules = config["rnn"]["train"]["ruleset"] + config["rnn"]["pretrain"]["ruleset"]

    hp = {
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
    nopretrain = "nopretrain" if nopretrain else "pretrain"
    run_model = main.load_model(
        f"../models/{activation}_{hidden_size}_{lr}__{freeze}_train_{nopretrain}.pth",
        hp,
        RNNLayer,
        device=device,
    )
    h = main.representation(run_model, all_rules)
    h_trans, explained_variance = main.compute_pca(h, n_components=n_components)
    # for key, value in h_trans.items():
    #     for i in range(value.shape[0]):
    #         h_trans[key][i] = value[i]
            # h_trans[key][i] = normalize_within_unit_volume(value[i])
    # ipdb.set_trace()
    # CHANGE HERE FOR OTHER TASKS!
    return h_trans[("AntiPerceptualDecisionMakingDelayResponseT", "stimulus")].detach().numpy(), explained_variance

def dsa_optimisation_compositionality(rank, n_delays, delay_interval, device):
    config = load_config("config.yaml")
    # Define parameters
    dt = config['simulations']['dt']
    num_steps = config['simulations']['num_steps']
    num_samples = config['simulations']['num_samples']
    lorenz_parameters = config['simulations']['lorenz_parameters']

    # Run simulations line
    simulations_line = simulation_line(num_steps, num_samples)

    # Run simulations curve
    simulations_curve = simulation_lorenz(dt, lorenz_parameters['one_attractor'][1], num_samples, num_steps)

    # Run simulations Pattern1
    simulations_pattern1 = simulation_lorenz(dt, lorenz_parameters['two_stable_attractors'][0], num_samples, num_steps)

    # Run simulations Pattern2
    simulations_pattern2 = simulation_lorenz(dt, lorenz_parameters['two_stable_attractors'][2], num_samples, num_steps)

    # Run simulations line-curve-line-curve
    combined_simulations_line_curve_line = combine_simulations([simulations_line, simulations_curve, np.flip(simulations_line, axis=0), np.flip(simulations_curve, axis=0)], method='attach')

    motif_basis = [simulations_line, simulations_curve, simulations_pattern1,simulations_pattern2,combined_simulations_line_curve_line]
    motif_names = ['Line', 'Curve', 'Pattern1', 'Pattern2','Line-Curve-Line-Curve']
    motif_dict = {motif_names[i]: motif_basis[i] for i in range(len(motif_basis))}
    all_simulations_length_3 = list(permutations(motif_names, 3))
    all_simulations_combined = {permutation: combine_simulations([motif_dict[permutation[0]], motif_dict[permutation[1]], motif_dict[permutation[2]]],method='attach') for permutation in all_simulations_length_3}

    model = list(all_simulations_combined.values())
    model_names = list(all_simulations_combined.keys())

    grouped_by_shared_elements = {i:[] for i in range(4)}
    for comp_motif_1 in model_names:
        for comp_motif_2 in model_names:
            if model_names.index(comp_motif_1) != model_names.index(comp_motif_2):
                set_1 = set(comp_motif_1)
                set_2 = set(comp_motif_2)
                grouped_by_shared_elements[len(set_1.intersection(set_2))].extend([(comp_motif_1, comp_motif_2)])
   
    similarities_grouped_by_shared_elements = {i:[] for i in range(4)}
    for key in grouped_by_shared_elements:
        for tuple1, tuple2 in grouped_by_shared_elements[key]:
            dsa = DSA.DSA(model[model_names.index(tuple1)], model[model_names.index(tuple2)], n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=True,iters=1000,lr=1e-2, device=device)
            similarities = dsa.fit_score()
            similarities_grouped_by_shared_elements[key].append(similarities)

    # compute median of similarities for each group and plot similarity vs number of shared elements
    median_similarities = {key: np.median(value) for key, value in similarities_grouped_by_shared_elements.items()}
    std_devs = {key: np.std(value) for key, value in similarities_grouped_by_shared_elements.items()}

    # Prepare data for plotting
    keys = list(median_similarities.keys())
    median_values = list(median_similarities.values())
    std_dev_values = list(std_devs.values())

    df = pd.DataFrame({'Number of shared elements': keys, 'Median similarity': median_values, 'Standard deviation': std_dev_values})
    
    # check if the directory exists
    if not os.path.exists('data/dsa_results'):
        os.makedirs('data/dsa_results')
    df.to_csv(f'data/dsa_results/{rank}_{n_delays}_{delay_interval}.csv')
    return 