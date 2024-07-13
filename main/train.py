""" Main training loop.
Copied from https://github.com/gyyang/multitask. Modified to work with pytorch instead of tensorflow framework. 
"""

from __future__ import division

import os
import warnings
import sys
import time
from collections import defaultdict

import torch
import math
import numpy as np
from neurogym import TrialEnv
from typing import List
from torch.cuda.amp import autocast, GradScaler
import torch
import time
import numpy as np
import main
import logging
import ipdb

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"

print_flag = False
######## mostly untouched ###############


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=os.path.join(log_dir, "training.log"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return logging


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


def get_default_hp(ruleset: List[str]):
    """Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    """
    basic_kwargs = {"dt": 20, "mode": "train", "rng": np.random.RandomState(0)}
    env = main.get_class_instance(ruleset[0], config=basic_kwargs)
    n_rule = len(ruleset)
    n_input, n_output = env.observation_space.shape[0] + n_rule, env.action_space.n
    hp = {
        "mode": "train",
        # batch size for training
        "batch_size_train": 128,
        # batch_size for testing
        "batch_size_test": 512,
        # input type: normal, multi
        "in_type": "normal",
        # Type of RNNs: LeakyRNN, LeakyGRU
        "rnn_type": "leaky_rnn",
        # whether rule and stimulus inputs are represented separately
        "use_separate_input": False,
        # Type of loss functions
        "loss_type": "crossentropy",
        # Optimizer
        "optimizer": "adam",
        # Type of activation runctions, relu, softplus, tanh, elu
        "activation": "relu",
        # Time constant (ms)
        "tau": 100,
        # discretization time step (ms)
        "dt": 20,
        # discretization time step/time constant
        "alpha": 0.2,
        # recurrent noise
        "sigma_rec": 0.05,
        # input noise
        "sigma_x": 0.01,
        # leaky_rec weight initialization, diag, randortho, randgauss
        "w_rec_init": "randortho",
        # a default weak regularization prevents instability
        "l1_h": 0,
        # l2 regularization on activity
        "l2_h": 0,
        # l2 regularization on weight
        "l1_weight": 0,
        # l2 regularization on weight
        "l2_weight": 0,
        # l2 regularization on deviation from initialization
        "l2_weight_init": 0,
        # proportion of weights to train, None or float between (0, 1)
        "p_weight_train": None,
        # Stopping performance
        "target_perf": 0.99,
        # number of rules
        "n_rule": n_rule,
        # first input index for rule units
        "rule_start": env.observation_space.shape[0],
        # number of input units
        "n_input": n_input,
        # number of output units
        "n_output": n_output,
        # number of recurrent units
        "n_rnn": 256,
        # number of input units
        "ruleset": ruleset,
        # name to save
        "save_name": "test",
        # learning rate
        "learning_rate": 0.001,
        # intelligent synapses parameters, tuple (c, ksi)
        "c_intsyn": 0,
        "ksi_intsyn": 0,
        "num_epochs": 10,
    }

    return hp


def set_hyperparameters(
    model_dir,
    hp=None,
    display_step=10,
    ruleset: List[str] = None,
    rule_trains: List[str] = None,
    rule_prob_map=None,
    seed=0,
    load_dir=None,
    trainables=None,
):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        num_epochs: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt    : Not implemented
        training configuration is stored at model_dir/hp.json
    """

    #     tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp["seed"] = seed
    hp["rng"] = np.random.RandomState(seed)
    hp["model_dir"] = model_dir
    hp["display_step"] = display_step
    hp["decay"] = math.exp(-hp["dt"] / hp["tau"])  # 1 - hp['dt']/hp['tau']
    hp["rule_trains"] = rule_trains
    hp["rules"] = rule_trains

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # penalty on deviation from initial weight
    if hp["l2_weight_init"] > 0:
        raise NotImplementedError

    # partial weight training
    if (
        "p_weight_train" in hp
        and (hp["p_weight_train"] is not None)
        and hp["p_weight_train"] < 1.0
    ):
        raise NotImplementedError

    if hp["optimizer"] == "adam":
        optimizer = torch.optim.Adam
    elif hp["optimizer"] == "sgd":
        optimizer = torch.optim.SGD
    else:
        raise NotImplementedError
    # Store results
    log = defaultdict(list)
    log["model_dir"] = model_dir

    return hp, log, optimizer  # , model


def train(run_model, optimizer, hp, log, name, freeze=False, retrain=False, rnn=False):

    # set up log
    logging = setup_logging(os.path.join(name, "logs"))

    start_epoch = 0

    # load checkpoint if there is any
    # TODO : adapt it to Mamba : How to save the model with the checkpoints?
    if not retrain:
        checkpoint_files = find_checkpoints(name)
        if checkpoint_files:
            latest_checkpoint = os.path.join(name, checkpoint_files[-1])
            checkpoint = torch.load(latest_checkpoint, map_location=run_model.device)
            run_model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            log = checkpoint["log"]
            print(f"Resuming training from epoch {start_epoch}")

    # freeze input weights or not
    # TODO : adapt it to Mamba
    if rnn:
        if freeze:
            optim = optimizer(
                [run_model.model.rnn.rnncell.weight_ih], lr=hp["learning_rate"]
            )
        else:
            optim = optimizer(run_model.model.parameters(), lr=hp["learning_rate"])
    else:
        if freeze:
            optim = optimizer(run_model.parameters(), lr=hp["learning_rate"])
        else:
            # only unfreeze embedding layer
            optim = optimizer(run_model.embedding.parameters(), lr=hp["learning_rate"])

    # if model loaded, load optim state dict
    if not retrain:
        if checkpoint_files:
            optim.load_state_dict(checkpoint["optimizer_state_dict"])

    # Create a GradScaler for mixed precision training
    scaler = GradScaler()

    dataloaders = {
        rule: main.get_dataloader(
            env=rule, batch_size=hp["batch_size_train"], num_workers=0, shuffle=True
        )
        for rule in hp["rule_trains"]
    }

    t_start = time.time()
    for epoch in range(start_epoch, hp["num_epochs"]):
        print(f"Epoch {epoch} started")
        epoch_loss = 0.0
        t_start_epoch = time.time()
        times_per_inputs = []
        times_between_inputs = []
        current_time = time.time()
        for rule in hp["rule_trains"]:
            for inputs, labels, mask in dataloaders[rule]["train"]:
                time_input = time.time()
                times_between_inputs.append(time.time() - current_time)
                inputs, labels, mask = (
                    inputs.permute(1, 0, 2).to(run_model.device, non_blocking=True),
                    labels.permute(1, 0)
                    .to(run_model.device, non_blocking=True)
                    .flatten()
                    .long(),
                    mask.permute(1, 0)
                    .to(run_model.device, non_blocking=True)
                    .flatten()
                    .long(),
                )
                optim.zero_grad(set_to_none=True)

                # autocast for mixed precision training
                with autocast():
                    c_lsq, c_reg, _, _, _ = run_model(inputs, labels, mask)
                    loss = c_lsq + c_reg if rnn else c_lsq

                # scale the loss and call backward() to create scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                epoch_loss += loss.item()
                times_per_inputs.append(
                    time.time() - time_input
                )  # time to process one input
                current_time = time.time()

        # doing evaluation
        log["trials"].append(epoch)
        log["times"].append(time.time() - t_start)
        # timing do_eval
        t_start_eval = time.time()
        log, logging = do_eval(run_model, log, logging, hp["rule_trains"], dataloaders)
        t_end_eval = time.time() - t_start_eval
        t_end_epoch = time.time() - t_start_epoch
        if log["perf_min"][-1] > hp["target_perf"]:
            break

        rule_train = hp["rule_trains"]
        if isinstance(rule_train, str):
            rule_name_print = rule_train
        else:
            rule_name_print = " & ".join(rule_train)

        logging.info(
            f"Time per input {np.median(times_per_inputs):0.6f} s | Time between inputs {np.median(times_between_inputs):0.6f} s | Time for evaluation {t_end_eval:0.2f} s"
        )
        logging.info(
            f"Training {rule_name_print} Epoch {epoch:7d} | Loss {epoch_loss:0.6f} | Perf {log['perf_avg'][-1]:0.2f} | Min {log['perf_min'][-1]:0.2f} | Time {t_end_epoch:0.2f} s"
        )

        start_save_time = time.time()
        # saving model checkpoint
        checkpoint_dir = name
        create_directory_if_not_exists(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_checkpoint.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": run_model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": epoch_loss,
                "log": log,
            },
            checkpoint_path,
        )
        end_save_time = time.time() - start_save_time
        logging.info(f"Time saving model :  {end_save_time:0.2f}s")

    logging.info("Optimization finished!")


def do_eval(run_model, log, logging, rule_train, dataloaders):
    hp = run_model.hp
    device = run_model.device

    # Lists to store times for all rules
    data_loading_times = []
    computation_times = []

    for rule_test in hp["rules"]:
        clsq_tmp, creg_tmp, perf_tmp = [], [], []
        dataloader = dataloaders[rule_test]["test"]

        for inputs, labels, mask in dataloader:
            # Record start time for data loading
            data_start_time = time.time()
            with torch.no_grad():
                inputs, labels, mask = (
                    inputs.permute(1, 0, 2).to(device, non_blocking=True),
                    labels.permute(1, 0).to(device, non_blocking=True).flatten().long(),
                    mask.permute(1, 0).to(device, non_blocking=True).flatten().long(),
                )
            # Record data loading time
            data_loading_time = time.time() - data_start_time
            data_loading_times.append(data_loading_time)

            # Record start time for computation
            computation_start_time = time.time()
            with torch.no_grad(), autocast():
                c_lsq, c_reg, y_hat_test, _, labels = run_model(inputs, labels, mask)

                clsq_tmp.append(c_lsq)  # Ensure this stays in float32 for stability
                creg_tmp.append(c_reg)

                perf_test = get_perf(y_hat_test, labels, mask)
                perf_tmp.append(perf_test)

            # Record computation time
            computation_time = time.time() - computation_start_time
            computation_times.append(computation_time)

        clsq_mean = torch.mean(torch.stack(clsq_tmp))
        creg_mean = torch.mean(torch.stack(creg_tmp))
        perf_mean = torch.mean(torch.tensor(perf_tmp))

        log["cost_" + rule_test].append(clsq_mean.item())
        log["creg_" + rule_test].append(creg_mean.item())
        log["perf_" + rule_test].append(perf_mean.item())

        logging.info(
            f"{rule_test:15s}| cost {clsq_mean.item():0.6f}| c_reg {creg_mean.item():0.6f} | perf {perf_mean.item():0.2f}"
        )
        sys.stdout.flush()

    perf_tests_mean = torch.mean(
        torch.tensor(
            [
                log["perf_" + r][-1]
                for r in (
                    rule_train if hasattr(rule_train, "__iter__") else [rule_train]
                )
            ]
        )
    )
    perf_tests_min = torch.min(
        torch.tensor(
            [
                log["perf_" + r][-1]
                for r in (
                    rule_train if hasattr(rule_train, "__iter__") else [rule_train]
                )
            ]
        )
    )

    log["perf_avg"].append(perf_tests_mean.item())
    log["perf_min"].append(perf_tests_min.item())

    # Calculate median times
    median_data_loading_time = np.median(data_loading_times)
    median_computation_time = np.median(computation_times)

    logging.info(
        f"Median data loading time: {median_data_loading_time:.2f}s, Median computation time: {median_computation_time:.2f}s"
    )

    return log, logging


def accuracy(logits, true_class_indices, mask):
    # Reshape logits to shape [(batch * images), classes]
    logits_flat = logits.view(-1, logits.size(-1))

    # Reshape true class indices to shape [(batch * images)]
    true_class_indices_flat = true_class_indices.flatten()

    # Reshape mask to shape [(batch * images)]
    mask_flat = mask.flatten()

    # put 1 when mask >1, 0 otherwise
    mask_flat = (mask_flat > 1).float()

    # Get the predicted classes by taking the argmax over the classes dimension
    predicted_classes = torch.argmax(logits_flat, dim=1)

    # Compare predicted classes with true class indices
    correct_predictions = (predicted_classes == true_class_indices_flat).float()

    # Apply the mask to weigh the correct predictions
    weighted_correct_predictions = correct_predictions * mask_flat

    # Calculate weighted accuracy
    total_weight = mask_flat.sum().item()
    weighted_accuracy = (
        weighted_correct_predictions.sum().item() / total_weight
        if total_weight > 0
        else 0.0
    )

    return weighted_accuracy


def get_perf(outputs, true_labels, mask):
    """Compute performance using PyTorch."""
    # Assuming outputs and true_labels are PyTorch tensors
    return accuracy(outputs, true_labels, mask)
