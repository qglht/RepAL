import main
from main import RNNLayer
from dsa_analysis import load_config, visualize
import torch
import pickle
import ipdb


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
    hp = {
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.000001,
        "l2_weight": 0.000001,
    }
    hp, log, optimizer = main.set_hyperparameters(
        model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=ruleset
    )


    if mode == "train":
        if no_pretraining:
            run_model = main.Run_Model(hp, RNNLayer, device)
            main.train(run_model, optimizer, hp, log, freeze=freeze)
            run_model.save(f"models/{activation}_{hidden_size}_{lr}__{freeze}_{mode}_nopretrain.pth")
        else: 
            run_model = main.load_model(
                f"models/{activation}_{hidden_size}_{lr}_pretrain.pth",
                hp,
                RNNLayer,
                device=device,
            )
            main.train(run_model, optimizer, hp, log, freeze=freeze)
            run_model.save(f"models/{activation}_{hidden_size}_{lr}__{freeze}_{mode}_pretrain.pth")
    elif mode == "pretrain":
        run_model = main.Run_Model(hp, RNNLayer, device)
        main.train(run_model, optimizer, hp, log, freeze=freeze)
        run_model.save(f"models/{activation}_{hidden_size}_{lr}_{mode}.pth")
    return run_model

def test_model(activation, hidden_size, lr, freeze, mode, no_pretraining, device):
    # Load configuration and set hyperparameters
    config = load_config("config.yaml")
    ruleset = config["rnn"][mode]["ruleset"]
    all_rules = config["rnn"]["train"]["ruleset"] + config["rnn"]["pretrain"]["ruleset"]
    hp = {
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.000001,
        "l2_weight": 0.000001,
    }
    hp, log, optimizer = main.set_hyperparameters(
        model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=ruleset
    )

    run_model = main.Run_Model(hp, RNNLayer, device)
    main.train(run_model, optimizer, hp, log, freeze=freeze)
    
    return run_model

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
        f"models/{activation}_{hidden_size}_{lr}__{freeze}_train_{nopretrain}.pth",
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