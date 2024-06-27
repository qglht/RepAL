from collections import OrderedDict
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ipdb
import torch
from torch import linalg as LA
from torch.cuda.amp import autocast
from main import get_dataloader, get_class_instance

#### Not modified for neurogym yet!!

def get_indexes(dt, timing, seq_length, h, rule):
    h_byepoch = OrderedDict()
    indexes = {key:None for key in timing.keys()}
    timing = {k: int(v/dt) for k,v in timing.items()}
    current_index = 0
    for e_name, e_time in timing.items():
        if current_index>=seq_length:
            return indexes
        else : 
            e_time_start = current_index
            e_time_end = current_index + e_time
            indexes[e_name] = (e_time_start, e_time_end)
            current_index += e_time
    t_start = int(500 / dt)  # Important: Ignore the initial transition
    for key, value in indexes.items():
        if value is not None:
            if value[1]>=t_start:
                value = (value[0]-t_start if value[0]-t_start>=0 else 0, value[1]-t_start)
            h_byepoch[(rule,key)] = h[value[0]:value[1], :, :]
    return h_byepoch 

def representation(model, rules):
   
    hp = model.hp
    if isinstance(rules, str):
        rules = [rules]
    elif rules is None:
        rules = hp["rules"]
    activations = OrderedDict()
    dataloaders = {rule: get_dataloader(env=rule, batch_size=hp["batch_size_train"], num_workers=16, shuffle=False, mode="test") for rule in rules}
    for rule in rules:
        # TODO : think about taking the good mode for the task otherwise the timing will be wrong
        # seq leng is the length of the cumulated timing
        env = get_class_instance(rule, config=hp)
        timing = env.timing
        seq_length = int(sum([v for k,v in timing.items()])/hp["dt"])
        # concatenate the activations for all trials
        for inputs, labels, mask in dataloaders[rule]["test"]:
            with torch.no_grad():
                inputs, labels, mask = inputs.permute(1, 0, 2).to(model.device, non_blocking=True), labels.permute(1, 0).to(model.device, non_blocking=True).flatten().long(), mask.permute(1, 0).to(model.device, non_blocking=True).flatten().long()
            with torch.no_grad(), autocast():
                _, _, _, h, _ = model(inputs, labels, mask)
            h_byepoch = get_indexes(hp['dt'], timing, seq_length, h, rule)
            for key, value in h_byepoch.items():
                if key not in activations:
                    activations[key] = value
                else:
                    activations[key] = torch.cat([activations[key], value], dim=1)

    return activations


def compute_pca(h, n_components=3):

    # only keep the stimulus period for dsa computation
    h = {k: v for k, v in h.items() if k[1] == 'stimulus'}

    # Concatenate across rules and epochs to create dataset
    data = torch.cat(list(h.values()), dim=0)
    data_2d = data.reshape(-1, data.shape[-1])

    # Reduce to 3 components for visualization
    U, S, V = LA.svd(data_2d, full_matrices=False)
    data_trans_2d = U[:, :n_components] @ torch.diag(S[:n_components])
    data_trans = data_trans_2d.reshape(data.shape[0], data.shape[1], n_components)

    # Compute explained variance for the reduction step
    explained_variance_ratio = (S[:n_components] ** 2) / (S**2).sum()

    # Package back to dictionary
    h_trans = OrderedDict()
    i_start = 0
    for key, val in h.items():
        i_end = i_start + val.shape[0]
        h_trans[key] = data_trans[i_start:i_end, :, :]
        i_start = i_end

    return h_trans, explained_variance_ratio.cumsum(dim=0)[-1].item()