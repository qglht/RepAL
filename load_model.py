import warnings

import main
from main import RNNLayer
from dsa_analysis import load_config
import torch
import os
import pandas as pd
from dsa_analysis import simulation_line, simulation_lorenz, combine_simulations, load_config
import DSA
import ipdb
import pandas as pd
import numpy as np
from itertools import permutations

path = "models_trained_8_models_per_group/tanh_128_0.001__True_train_nopretrain/epoch_19_checkpoint.pth"

def load_model(path):
    model = torch.load(path)
    return model

config = load_config("config.yaml")
all_rules = config['all_rules']
hp = {
    "activation": "tanh",
    "n_rnn": 128,
    "learning_rate": 0.001,
    "l2_h": 0.000001,
    "l2_weight": 0.000001,
}
hp, log, optimizer = main.set_hyperparameters(
    model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=all_rules
)
ipdb.set_trace()
run_model = main.load_model(
        path,
        hp,
        RNNLayer,
        device="cpu",
    )


    