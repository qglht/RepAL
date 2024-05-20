""" Main training loop.
Copied from https://github.com/gyyang/multitask. Modified to work with pytorch instead of tensorflow framework. 
"""

from __future__ import division

import sys
import time
from collections import defaultdict

import torch
import math
import numpy as np
import ipdb
from . import task


print_flag = False


def _gen_feed_dict(trial, hp, device):
    # Ensure all data is already on the correct device
    trial.x = torch.as_tensor(trial.x, device=device)
    trial.y = torch.as_tensor(trial.y, device=device)
    trial.c_mask = torch.as_tensor(trial.c_mask, device=device)
    trial.y_loc = torch.as_tensor(trial.y_loc, device=device)

    n_time, batch_size = trial.x.shape[:2]

    if hp["in_type"] == "multi":
        new_shape = [n_time, batch_size, hp["rule_start"] * hp["n_rule"]]
        x = torch.zeros(new_shape, dtype=torch.float32, device=device)

        for i in range(batch_size):
            ind_rule = torch.argmax(trial.x[0, i, hp["rule_start"] :]).item()
            i_start = ind_rule * hp["rule_start"]
            x[:, i, i_start : i_start + hp["rule_start"]] = trial.x[
                :, i, : hp["rule_start"]
            ]

        trial.x = x

    trial.c_mask = trial.c_mask.view(n_time, batch_size, -1)
    return trial


def generate_trials(rule, hp, mode, batch_size, device):
    if batch_size is None:
        trial = task.generate_trials(rule, hp, mode=mode)
    else:
        trial = task.generate_trials(rule, hp, mode=mode, batch_size=batch_size)
    trial = _gen_feed_dict(trial, hp, device)
    return trial


######## mostly untouched ###############


def get_default_hp(ruleset):
    """Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    """
    num_ring = task.get_num_ring(ruleset)
    n_rule = task.get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    hp = {
        # batch size for training
        "batch_size_train": 64,
        # batch_size for testing
        "batch_size_test": 512,
        # input type: normal, multi
        "in_type": "normal",
        # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
        "rnn_type": "LeakyRNN",
        # whether rule and stimulus inputs are represented separately
        "use_separate_input": False,
        # Type of loss functions
        "loss_type": "lsq",
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
        "target_perf": 0.95,
        # number of units each ring
        "n_eachring": n_eachring,
        # number of rings
        "num_ring": num_ring,
        # number of rules
        "n_rule": n_rule,
        # first input index for rule units
        "rule_start": 1 + num_ring * n_eachring,
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
    }

    return hp


def display_rich_output(model, step, log, model_dir):
    """Display step by step outputs during training."""
    variance._compute_variance_bymodel(model)
    rule_pair = ["contextdm1", "contextdm2"]
    save_name = "_atstep" + str(step)
    title = "Step " + str(step) + " Perf. {:0.2f}".format(log["perf_avg"][-1])
    variance.plot_hist_varprop(
        model_dir, rule_pair, figname_extra=save_name, title=title
    )
    plt.close("all")


def set_hyperparameters(
    model_dir,
    hp=None,
    max_steps=1e7,
    display_step=500,
    ruleset="mante",
    rule_trains=None,
    rule_prob_map=None,
    seed=0,
    rich_output=False,
    load_dir=None,
    trainables=None,
):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
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
    hp["max_steps"] = max_steps
    hp["display_step"] = display_step
    hp["rich_output"] = rich_output
    hp["decay"] = math.exp(-hp["dt"] / hp["tau"])  # 1 - hp['dt']/hp['tau']

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp["rule_trains"] = task.rules_dict[ruleset]
    else:
        hp["rule_trains"] = rule_trains
    hp["rules"] = hp["rule_trains"]

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp["rule_probs"] = None
    if hasattr(hp["rule_trains"], "__iter__"):
        # Set default as 1.
        rule_prob = np.array([rule_prob_map.get(r, 1.0) for r in hp["rule_trains"]])
        hp["rule_probs"] = list(rule_prob / np.sum(rule_prob))

    #     tools.save_hp(hp, model_dir)     # saving model: not implemented

    #     ##### Build the model #####
    #     model = Model(model_dir, hp=hp)  # model is defined outside.

    if print_flag:  # Display hp
        for key, val in hp.items():
            print("{:20s} = ".format(key) + str(val))

    if load_dir is not None:
        raise NotImplementedError  # loading saved model: not implemented

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


def train(run_model, optimizer, hp, log, freeze=False):

    step = 0
    t_start = time.time()
    losses = []
    loss_change_threshold = 1e-4  # Threshold for change in loss to consider stopping
    if freeze:
        optim = optimizer(
            [run_model.model.rnn.rnncell.weight_ih], lr=hp["learning_rate"]
        )
    else:
        optim = optimizer(run_model.model.parameters(), lr=hp["learning_rate"])
    while step * hp["batch_size_train"] <= hp["max_steps"]:
        try:
            # Validation
            if step % hp["display_step"] == 0:
                log["trials"].append(step * hp["batch_size_train"])
                log["times"].append(time.time() - t_start)
                log = do_eval(run_model, log, hp["rule_trains"])
                # if loss is not decreasing anymore, stop training
                # check if minimum performance is above target
                if log["perf_min"][-1] > hp["target_perf"]:
                    # Check if the average decrease in loss is below a certain threshold
                    recent_losses = losses[-hp["batch_size_train"] * 2 :]
                    avg_loss_change = np.mean(np.diff(recent_losses))
                    if abs(avg_loss_change) < loss_change_threshold:
                        print(
                            "Perf reached the target: {:0.2f}".format(hp["target_perf"])
                        )
                        break

                if hp["rich_output"]:
                    display_rich_output(run_model, step, log, hp["model_dir"])

            # Training
            rule_train_now = hp["rng"].choice(hp["rule_trains"], p=hp["rule_probs"])

            optim.zero_grad()
            c_lsq, c_reg, _, _, _ = run_model(
                rule=rule_train_now, batch_size=hp["batch_size_train"]
            )
            loss = c_lsq + c_reg
            losses.append(loss.item())
            loss.backward()
            optim.step()
            step += 1

        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            break

    print("Optimization finished!")


def do_eval(run_model, log, rule_train):
    """Do evaluation using entirely PyTorch operations to ensure GPU utilization."""
    hp = run_model.hp
    if isinstance(rule_train, str):
        rule_name_print = rule_train
    else:
        rule_name_print = " & ".join(rule_train)

    print(
        f"Trial {log['trials'][-1]:7d} | Time {log['times'][-1]:0.2f} s | Now training {rule_name_print}"
    )

    for rule_test in hp["rules"]:
        n_rep = 16
        batch_size_test_rep = hp["batch_size_test"] // n_rep
        clsq_tmp, creg_tmp, perf_tmp = [], [], []

        for i_rep in range(n_rep):
            with torch.no_grad():
                c_lsq, c_reg, y_hat_test, _, trial = run_model(
                    rule=rule_test, batch_size=batch_size_test_rep
                )

            # Store costs directly as tensors to avoid multiple GPU to CPU transfers
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)

            # Calculate performance using PyTorch
            perf_test = get_perf(y_hat_test, trial.y_loc)
            perf_test = perf_test.mean()
            perf_tmp.append(perf_test)

        # Convert lists of tensors to single tensors and compute mean
        clsq_mean = torch.mean(torch.stack(clsq_tmp))
        creg_mean = torch.mean(torch.stack(creg_tmp))
        perf_mean = torch.mean(torch.stack(perf_tmp))

        # Append to log dictionary (transfer to CPU and convert to numpy for logging purposes only)
        log["cost_" + rule_test].append(clsq_mean.item())
        log["creg_" + rule_test].append(creg_mean.item())
        log["perf_" + rule_test].append(perf_mean.item())

        print(
            f"{rule_test:15s}| cost {clsq_mean.item():0.6f}| c_reg {creg_mean.item():0.6f} | perf {perf_mean.item():0.2f}"
        )
        sys.stdout.flush()

    # Calculate average and minimum performance across rules being trained
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

    return log


##################     copied from network.py     ######################


def popvec(y):
    """Population vector readout using PyTorch."""
    # Assuming y is a PyTorch tensor
    prefs = torch.linspace(0, 2 * np.pi, y.shape[-1], device=y.device)  # preferences
    y_sum = y.sum(dim=-1)
    y_cos = torch.sum(y * torch.cos(prefs), dim=-1) / y_sum
    y_sin = torch.sum(y * torch.sin(prefs), dim=-1) / y_sum
    loc = torch.atan2(y_sin, y_cos)
    return loc % (2 * np.pi)


def get_perf(y_hat, y_loc):
    """Compute performance using PyTorch."""
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    fixating = y_hat_fix > 0.5

    original_dist = torch.abs(y_loc - y_hat_loc)
    dist = torch.min(original_dist, 2 * np.pi - original_dist)
    corr_loc = dist < 0.2 * np.pi

    should_fix = y_loc < 0

    perf = (should_fix * fixating) + (~should_fix * corr_loc * (~fixating))
    perf = perf.float()
    return perf