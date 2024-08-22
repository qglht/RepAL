from collections import OrderedDict
from tracemalloc import start
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ipdb
from sklearn.impute import SimpleImputer
import torch
from torch import Value, linalg as LA
from main import get_dataloader, get_class_instance
import copy

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


def representation(model, rules, rnn=True, rnn_vs_mamba=False):
    hp = model.hp
    rules = [rules] if isinstance(rules, str) else (rules or hp["rules"])
    activations = OrderedDict()

    # Batch-loading optimizations:
    with torch.no_grad():
        for rule in rules:
            env = get_class_instance(rule, config=hp)
            timing = env.timing
            seq_length = int(sum(timing.values()) / hp["dt"])
            if not rnn_vs_mamba:
                dataloader = get_dataloader(
                    env=rule,
                    batch_size=hp["batch_size_train"],
                    num_workers=0,
                    shuffle=False,
                    mode="test",
                )[
                    "test"
                ]  # Directly access the test dataloader
            else:
                # For RNN vs MAMBA comparison, use a smaller batch size and the same bs
                dataloader = get_dataloader(
                    env=rule,
                    batch_size=64,
                    num_workers=0,
                    shuffle=False,
                    mode="test",
                )["test"]

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

                if rnn:
                    _, _, _, h, _ = model(inputs, labels, mask)
                else:

                    h = model.get_activations(inputs)

                # move to cpu
                h = h.cpu()

                h_byepoch = get_indexes(hp["dt"], timing, seq_length, h, rule)
                for key, value in h_byepoch.items():
                    activations.setdefault(key, []).append(
                        value
                    )  # Accumulate in a list

            del dataloader
            torch.cuda.empty_cache()  # If using GPU

    # Merge accumulated activations:
    for key, values in activations.items():
        activations[key] = torch.cat(
            values, dim=1
        )  # Concatenate the tensors along the batch dimension
    # only return activations for which key[1] == 'stimulus'
    activations_stimulus = None
    try:
        activations_stimulus = activations[
            ("AntiPerceptualDecisionMakingDelayResponseT", "stimulus")
        ]
    except KeyError:
        activations_stimulus = activations[("AntiGoNogoDelayResponseT", "stimulus")]

    # # Center and standardize the activations
    # flattened_activations = activations_stimulus.reshape(
    #     -1, activations_stimulus.shape[-1]
    # )
    # mean_activations = torch.mean(flattened_activations, dim=0)
    # std_activations = torch.std(flattened_activations, dim=0)

    # # Handle potential division by zero
    # std_activations[std_activations == 0] = 1.0

    # activations_stimulus = (activations_stimulus - mean_activations) / std_activations
    return activations_stimulus


def compute_pca(h, n_components=3):
    h = {k: v for k, v in h.items() if k[1] == "stimulus"}
    try:
        h = h[("AntiPerceptualDecisionMakingDelayResponseT", "stimulus")]
    except KeyError:
        h = h[("AntiGoNogoDelayResponseT", "stimulus")]
    data = h
    # ipdb.set_trace()
    # data = torch.cat(list(h.values()), dim=0)
    data_2d = data.reshape(-1, data.shape[-1])

    # Using PCA directly for dimensionality reduction:
    pca = PCA(n_components=n_components)
    data_trans_2d = torch.tensor(pca.fit_transform(data_2d.cpu().numpy())).to(
        data_2d.device
    )  # Convert back to PyTorch tensor

    # Compute explained variance ratio using PCA
    explained_variance_ratio = pca.explained_variance_ratio_.sum()

    data_trans = data_trans_2d.reshape(data.shape[0], data.shape[1], n_components)

    # # Package back to dictionary
    # h_trans = OrderedDict()
    # i_start = 0
    # for key, val in h.items():
    #     i_end = i_start + val.shape[0]
    #     h_trans[key] = data_trans[i_start:i_end, :, :]
    #     i_start = i_end

    return data_trans, explained_variance_ratio


def compute_common_pca(h_list, n_components=3):
    # here h is a list of dictionaries, each containing activations for a different rule
    print(f"len(h_list): {len(h_list)}")
    h_list = [h if torch.is_tensor(h) else torch.tensor(h) for h in h_list]
    # print the shapes of the tensors in h_list
    for i, h in enumerate(h_list):
        print(f"h_list[{i}].shape: {h.shape}")

    # equalize n_neurons

    # Define the target number of neurons (minimum n_neurons)
    min_neurons = min(h_list[0].shape[2], h_list[1].shape[2])
    max_neurons = max(h_list[0].shape[2], h_list[1].shape[2])

    if min_neurons < max_neurons:
        # index to reduce
        index_to_reduce = 0 if h_list[0].shape[2] > min_neurons else 1
        # find the h_list with the max number of neurons
        h_to_reduce = h_list[index_to_reduce]

        # Flatten the first two dimensions (n_time and n_batch) for PCA
        flattened_h1 = h_to_reduce.reshape(-1, h_to_reduce.shape[2])

        # Apply PCA to reduce h_list[1] to min_neurons
        pca = PCA(n_components=min_neurons)
        reduced_h1 = pca.fit_transform(flattened_h1)

        # Reshape back to original n_time and n_batch dimensions
        reduced_h1 = torch.tensor(reduced_h1).reshape(
            h_to_reduce.shape[0], h_to_reduce.shape[1], min_neurons
        )

        # Replace the original h_list[1] with the reduced version
        h_list[index_to_reduce] = reduced_h1

        # # print the shapes of the tensors in h_list transformed
        for i, h in enumerate(h_list):
            print(f"h_list[{i}].shape after Transformation: {h.shape}")

    data = torch.cat(h_list, dim=1)

    # mean center and std data
    mean_activations = torch.mean(data, dim=0)
    std_activations = torch.std(data, dim=0)
    # Handle potential division by zero
    std_activations[std_activations == 0] = 1.0
    data = (data - mean_activations) / std_activations

    data_2d = data.reshape(-1, data.shape[-1])

    # Convert to numpy array for PCA
    data_2d_np = data_2d.cpu().numpy()

    # Handle NaN values by imputing with the mean of the column
    imputer = SimpleImputer(strategy="mean")
    data_2d_np = imputer.fit_transform(data_2d_np)

    # Perform PCA on the imputed data
    pca = PCA(n_components=n_components)
    data_trans_2d_np = pca.fit_transform(data_2d_np)

    # Convert the transformed data back to a PyTorch tensor on the original device
    data_trans_2d = torch.tensor(data_trans_2d_np)
    # Compute explained variance ratio using PCA
    explained_variance_ratio = pca.explained_variance_ratio_.sum()

    data_trans = data_trans_2d.reshape(data.shape[0], data.shape[1], n_components)
    # Package back to list of activations as initial h_list
    pca_h_list = []
    start = 0
    for i in range(len(h_list)):
        print(f"i: {i}")
        end = start + h_list[i].shape[1]
        curve = data_trans[:, start:end, :].cpu()
        curve = curve.detach().numpy()
        curve_reduced = copy.deepcopy(curve)
        pca_h_list.append(curve_reduced)
        start = end

    # h_trans = OrderedDict()
    # i_start = 0
    # for key, val in h.items():
    #     i_end = i_start + val.shape[0]
    #     h_trans[key] = data_trans[i_start:i_end, :, :]
    #     i_start = i_end

    return pca_h_list, explained_variance_ratio
