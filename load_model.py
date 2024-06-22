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
import similarity
import ipdb

def same_order(comp_motif_1, comp_motif_2)-> bool:
    return len([i for i in range(len(comp_motif_1)) if comp_motif_1[i] == comp_motif_2[i]])

def find_checkpoints(name):
    # Find the latest checkpoint file
    checkpoint_dir = name
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('_checkpoint.pth')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1]))
    return checkpoint_files

def initialize_model(rnn_type, activation, hidden_size, lr, batch_size, device):
    config = load_config("config.yaml")
    all_rules = config['all_rules']
    hp = {
        "rnn_type": rnn_type,
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.0001,
        "l2_weight": 0.0001,
        "num_epochs": 50,
        "batch_size_train":batch_size,
        "mode": "test", 
    }
    hp, log, optimizer = main.set_hyperparameters(
            model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=all_rules
        )
    run_model = main.Run_Model(hp, RNNLayer, device)

    return run_model, hp
    
def corresponding_training_time(n, p):
    # Find the argmin over j for each i
    return [min(range(p), key=lambda j: abs(int(100 * i / (n - 1)) - int(100 * j / (p - 1)))) for i in range(n)]

def get_curves(model, rules, components):
    h = main.representation(model, rules)
    h_trans, explained_variance = main.compute_pca(h, n_components=components)
    return h_trans[("AntiPerceptualDecisionMakingDelayResponseT", "stimulus")].detach().numpy()


config = load_config("config.yaml")
all_rules = config["all_rules"]
rnn_type = "leaky_rnn"
activation = "tanh"
hidden_size = 128
lr = 0.0001
batch_size = 128
device = "cpu"

# paths for checkpoints
path_train_folder1 = os.path.join(f"models_trained_8_models_per_group/tanh_128_0.0001__True_train_nopretrain")
path_train_folder2 = os.path.join(f"models_trained_8_models_per_group/tanh_128_0.0001__False_train_nopretrain")

# initialize model architectures
run_model1, hp1 = initialize_model(rnn_type, activation, hidden_size, lr, batch_size, device)
run_model2, hp2 = initialize_model(rnn_type, activation, hidden_size, lr, batch_size, device)

# get checkpoints in train path
checkpoint_files_1 = find_checkpoints(path_train_folder1)
checkpoint_files_2 = find_checkpoints(path_train_folder2)

# group models and establish correspondancy between epochs
models_to_compare = []
dissimilarities_over_learning = {"cka":[],"dsa":[],"procrustes":[]}
cka_measure = similarity.make("measure.sim_metric.cka-angular-score")
procrustes_measure = similarity.make("measure.netrep.procrustes-angular-score")
if checkpoint_files_1 and checkpoint_files_2:
    # get the models to compare
    index_epochs = corresponding_training_time(len(checkpoint_files_1), len(checkpoint_files_2))
    ipdb.set_trace()
    for epoch in index_epochs:
        checkpoint1 = torch.load(os.path.join(path_train_folder1, checkpoint_files_1[index_epochs.index(epoch)]), map_location=device)
        run_model1.load_state_dict(checkpoint1['model_state_dict'])
        print(f"checkpoint 1 {checkpoint1['log']}")
        checkpoint2 = torch.load(os.path.join(path_train_folder2, checkpoint_files_2[index_epochs[epoch]]), map_location=device)
        run_model2.load_state_dict(checkpoint2['model_state_dict'])
        print(f"checkpoint 2 {checkpoint2['log']}")
        models_to_compare.extend([(run_model1, run_model2)])

    # compute the curves for models and dissimilarities
    curves = [(get_curves(tuple_model[0], all_rules, components=15), get_curves(tuple_model[1], all_rules, components=15)) for tuple_model in models_to_compare]
    for epoch in index_epochs:
        dissimilarities_over_learning["cka"].append(1-cka_measure(curves[epoch][0], curves[epoch][1]))
        dissimilarities_over_learning["procrustes"].append(1-procrustes_measure(curves[epoch][0], curves[epoch][1]))
        dsa_comp = DSA.DSA(
            curves[epoch][0], curves[epoch][1],
            n_delays=config["dsa"]["n_delays"],
            rank=config["dsa"]["rank"],
            delay_interval=config["dsa"]["delay_interval"],
            verbose=True,
            iters=1000,
            lr=1e-2,
        )
        dissimilarities_over_learning["dsa"].append(dsa_comp.fit_score())
ipdb.set_trace()


