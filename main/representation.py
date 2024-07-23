from collections import OrderedDict
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ipdb
import torch
from torch import Value, linalg as LA
from main import get_dataloader, get_class_instance

# import PCA
from sklearn.decomposition import PCA

#### Not modified for neurogym yet!!


def get_indexes(dt, timing, seq_length, h, rule):
    h_byepoch = OrderedDict()
    indexes = {key: None for key in timing.keys()}
    timing = {k: int(v / dt) for k, v in timing.items()}
    current_index = 0
    for e_name, e_time in timing.items():
        if current_index >= seq_length:
            return indexes
        else:
            e_time_start = current_index
            e_time_end = current_index + e_time
            indexes[e_name] = (e_time_start, e_time_end)
            current_index += e_time
    t_start = int(500 / dt)  # Important: Ignore the initial transition
    for key, value in indexes.items():
        if value is not None:
            if value[1] >= t_start:
                value = (
                    value[0] - t_start if value[0] - t_start >= 0 else 0,
                    value[1] - t_start,
                )
            h_byepoch[(rule, key)] = h[value[0] : value[1], :, :]
    return h_byepoch


def representation(model, rules, rnn=True):
    hp = model.hp
    rules = [rules] if isinstance(rules, str) else (rules or hp["rules"])
    activations = OrderedDict()

    # Batch-loading optimizations:
    with torch.no_grad():
        for rule in rules:
            env = get_class_instance(rule, config=hp)
            timing = env.timing
            seq_length = int(sum(timing.values()) / hp["dt"])

            dataloader = get_dataloader(
                env=rule,
                batch_size=hp["batch_size_train"],
                num_workers=0,
                shuffle=False,
                mode="test",
            )[
                "test"
            ]  # Directly access the test dataloader

            for inputs, labels, mask in dataloader:
                if rnn:
                    inputs, labels, mask = (
                        inputs.permute(1, 0, 2),
                        labels.permute(1, 0),
                        mask.permute(1, 0),
                    )
                inputs = inputs.to(model.device, non_blocking=True)
                labels = labels.to(model.device, non_blocking=True).flatten().long()
                mask = mask.to(model.device, non_blocking=True).flatten().long()

                _, _, _, h, _ = model(inputs, labels, mask)
                h_byepoch = get_indexes(hp["dt"], timing, seq_length, h, rule)
                for key, value in h_byepoch.items():
                    activations.setdefault(key, []).append(
                        value
                    )  # Accumulate in a list

    # Merge accumulated activations:
    for key, values in activations.items():
        activations[key] = torch.cat(
            values, dim=1
        )  # Concatenate the tensors along the batch dimension
    return activations


def compute_pca(h, n_components=3):
    h = {k: v for k, v in h.items() if k[1] == "stimulus"}
    data = torch.cat(list(h.values()), dim=0)
    data_2d = data.reshape(-1, data.shape[-1])

    # Using PCA directly for dimensionality reduction:
    pca = PCA(n_components=n_components)
    data_trans_2d = torch.tensor(pca.fit_transform(data_2d.cpu().numpy())).to(
        data_2d.device
    )  # Convert back to PyTorch tensor

    # Compute explained variance ratio using PCA
    explained_variance_ratio = pca.explained_variance_ratio_.sum()

    data_trans = data_trans_2d.reshape(data.shape[0], data.shape[1], n_components)

    # Package back to dictionary
    h_trans = OrderedDict()
    i_start = 0
    for key, val in h.items():
        i_end = i_start + val.shape[0]
        h_trans[key] = data_trans[i_start:i_end, :, :]
        i_start = i_end

    return h_trans, explained_variance_ratio
