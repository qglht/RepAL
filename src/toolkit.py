import warnings

import main
from main import RNNLayer
from dsa_analysis import load_config, visualize
import torch
import pickle
import ipdb
import os

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
            if (name+".pth") in os.listdir("models"):
                return
            else:
                run_model = main.Run_Model(hp, RNNLayer, device)
                main.train(run_model, optimizer, hp, log, name, freeze=freeze)
                run_model.save(name+".pth")
        else: 
            name = os.path.join("models", model_name + f"__{freeze}_train_pretrain") 
            if (name+".pth") in os.listdir("models"):
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
        if (name+".pth") in os.listdir("models"):
            return
        else:
            run_model = main.Run_Model(hp, RNNLayer, device)
            main.train(run_model, optimizer, hp, log, name, freeze=freeze)
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
    h = main.representation(run_model, config["rnn"]["train"]["ruleset"])
    h_trans, explained_variance = main.compute_pca(h, n_components=n_components)
    # for key, value in h_trans.items():
    #     for i in range(value.shape[0]):
    #         h_trans[key][i] = value[i]
            # h_trans[key][i] = normalize_within_unit_volume(value[i])
    # ipdb.set_trace()
    # CHANGE HERE FOR OTHER TASKS!
    return h_trans[("AntiPerceptualDecisionMakingDelayResponseT", "stimulus")].detach().numpy(), explained_variance