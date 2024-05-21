from collections import OrderedDict
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ipdb
import torch
from torch import linalg as LA

#### Not modified for neurogym yet!!

def representation(model, rules):
    h_byepoch = OrderedDict()
    hp = model.hp
    if isinstance(rules, str):
        rules = [rules]
    elif rules is None:
        rules = hp["rules"]

    for rule in rules:
        _, _, _, h, trial = model(rule=rule, mode="test")
        t_start = int(500 / hp["dt"])  # Important: Ignore the initial transition
        h = h[t_start:, :, :]
        for e_name, e_time in trial.epochs.items():
            if "fix" in e_name:
                continue
            # Take epoch
            e_time_start = e_time[0] - 1 if e_time[0] > 0 else 0
            h_byepoch[(rule, e_name)] = h[e_time_start : e_time[1], :, :]

    return h_byepoch


def compute_pca(h):
    # Concatenate across rules and epochs to create dataset
    data = torch.cat(list(h.values()), dim=0)
    data_2d = data.reshape(-1, data.shape[-1])

    # Reduce to 3 components for visualization
    U, S, V = LA.svd(data_2d, full_matrices=False)
    data_trans_2d = U[:, :3] @ torch.diag(S[:3])
    data_trans = data_trans_2d.reshape(data.shape[0], data.shape[1], 3)

    # Compute explained variance for the reduction step
    explained_variance_ratio = (S[:3] ** 2) / (S**2).sum()

    # Package back to dictionary
    h_trans = OrderedDict()
    i_start = 0
    for key, val in h.items():
        i_end = i_start + val.shape[1]
        h_trans[key] = data_trans[i_start:i_end, :, :]
        i_start = i_end

    return h_trans, explained_variance_ratio.cumsum(dim=0)[-1].item()