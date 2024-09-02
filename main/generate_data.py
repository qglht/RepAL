import neurogym as ngym
import numpy as np
import torch
import importlib
import pickle


def get_class_instance(class_name, **kwargs):
    module = importlib.import_module("main")
    class_ = getattr(module, class_name)
    instance = class_(**kwargs)
    return instance


def swap_axes(inputs, targets):
    return np.swapaxes(inputs, 0, 1), np.swapaxes(targets, 0, 1)


def gen_feed_data(inputs, labels, env, hp):
    n_samples, n_time = inputs.shape[:2]

    new_shape = (n_samples, n_time, hp["rule_start"] + hp["n_rule"])
    x = np.zeros(new_shape, dtype=np.float32)  # np.zeros instead of torch.zeros
    ind_rule = hp["ruleset"].index(env)
    x[:, :, : hp["rule_start"]] = inputs  # Slicing works the same
    x[:, :, hp["rule_start"] + ind_rule] = 1  # Setting values
    inputs = x

    return inputs, labels


def create_mask(inputs):
    n_sample, n_time, _ = inputs.shape
    zero_mask = np.all(inputs == 0, axis=2)
    mask = np.where(zero_mask, 5.0, 1.0).astype(np.float32)
    for b in range(n_sample):
        count = 0
        for i in range(n_time):
            if mask[b, i] == 5:
                if count < 4:
                    mask[b, i] = 2 + count
                count += 1
            else:
                count = 0
    return mask


def generate_data(env, hp, mode, seq_len=400, num_pregenerated=100000):
    env_instance = get_class_instance(env, config=hp)
    if mode == "test":
        timing = env_instance.timing
        seq_len = int(sum([v for k, v in timing.items()]) / hp["dt"])
    dataset = ngym.Dataset(env_instance, batch_size=32, seq_len=seq_len)
    inputs_list = []
    targets_list = []

    for _ in range(num_pregenerated // 32):
        input_sample, target_sample = dataset()
        inputs_list.append(input_sample)
        targets_list.append(target_sample)

    # Handling the remaining samples
    remaining_samples = num_pregenerated % 32
    if remaining_samples > 0:
        input_sample, target_sample = dataset()
        inputs_list.append(input_sample[:, :remaining_samples])
        targets_list.append(target_sample[:, :remaining_samples])
    inputs = np.concatenate(inputs_list, axis=1)
    targets = np.concatenate(targets_list, axis=1)

    # Swap axes after gathering all samples
    inputs, targets = swap_axes(inputs, targets)

    # Create masks
    masks = create_mask(inputs)

    # Generate feed data
    inputs, targets = gen_feed_data(inputs, targets, env, hp)

    # Convert inputs, targets, masks to torch tensors
    inputs, targets, masks = (
        torch.tensor(inputs).to(torch.float32),
        torch.tensor(targets).to(torch.float32),
        torch.tensor(masks).to(torch.float32),
    )
    dataset = {"inputs": inputs, "targets": targets, "masks": masks}

    with open(f"data/{env}_{mode}.pkl", "wb") as f:
        pickle.dump(dataset, f)


def generate_data_vis(env, hp, mode, seq_len=400, num_pregenerated=100000):
    env_instance = get_class_instance(env, config=hp)
    print(env_instance)
    if mode == "test":
        timing = env_instance.timing
        print(timing)
        seq_len = int(sum([v for k, v in timing.items()]) / hp["dt"])
    dataset = ngym.Dataset(env_instance, batch_size=32, seq_len=seq_len)
    inputs_list = []
    targets_list = []

    for _ in range(num_pregenerated // 32):
        input_sample, target_sample = dataset()
        inputs_list.append(input_sample)
        targets_list.append(target_sample)

    # Handling the remaining samples
    remaining_samples = num_pregenerated % 32
    if remaining_samples > 0:
        input_sample, target_sample = dataset()
        inputs_list.append(input_sample[:, :remaining_samples])
        targets_list.append(target_sample[:, :remaining_samples])
    inputs = np.concatenate(inputs_list, axis=1)
    targets = np.concatenate(targets_list, axis=1)

    # Swap axes after gathering all samples
    inputs, targets = swap_axes(inputs, targets)

    # Create masks
    masks = create_mask(inputs)

    # Generate feed data
    inputs, targets = gen_feed_data(inputs, targets, env, hp)

    # Convert inputs, targets, masks to torch tensors
    inputs, targets, masks = (
        torch.tensor(inputs).to(torch.float32),
        torch.tensor(targets).to(torch.float32),
        torch.tensor(masks).to(torch.float32),
    )
    dataset = {"inputs": inputs, "targets": targets, "masks": masks}

    return dataset
