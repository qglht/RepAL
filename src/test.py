from dsa_analysis import load_config, visualize
import torch
from src.toolkit import compute_dissimilarity, train_model
import DSA
# import similarity
import pickle
import numpy as np
import ipdb
import matplotlib.pyplot as plt
import multiprocessing
import seaborn as sns
import main
from main.dataset import Dataset, get_class_instance


if __name__ == "__main__":

    multiprocessing.set_start_method(
        "spawn", force=True)
    # )  # Set multiprocessing to use 'spawn'
    # config = load_config("config.yaml")

    # # Create a list of all tasks to run
    # tasks = []
    # num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    # devices = (
    #     [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    #     if num_gpus > 0
    #     else [torch.device("cpu")]
    # )
    # print(f'devices used : {devices}')

    # i = 0  # Index to cycle through available devices

    # device = devices[0]
    train_model("softplus", 128, 0.001, False, "train", False, "cpu")
    # curves_frozen = []
    # curves_frozen_names = []
    # curves_unfrozen = []
    # curves_unfrozen_names = []
    # explained_variances_frozen = []
    # explained_variances_unfrozen = []
    # dissimilarities = {"within_unfrozen": {}, "within_frozen": {}, "across": []}
    # for activation in config["rnn"]["parameters"]["activations"]:
    #     for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
    #         for lr in config["rnn"]["parameters"]["learning_rate"]:
    #             for freeze in config["rnn"]["parameters"]["freeze"]:
    #                 # curve, explained_variance = compute_dissimilarity(
    #                 #     activation, hidden_size, lr, freeze, "cpu"
    #                 # )
    #                 device = devices[i % len(devices)]  # Cycle through available devices
    #                 train_model(activation, hidden_size, lr, freeze, "pretrain", True, "cpu")
    #                 # if freeze:
    #                 #     curves_frozen.append(curve)
    #                 #     curves_frozen_names.append(
    #                 #         f"{activation}_{hidden_size}_{lr}"
    #                 #     )
    #                 #     explained_variances_frozen.append(explained_variance)
    #                 # else:
    #                 #     curves_unfrozen.append(curve)
    #                 #     curves_unfrozen_names.append(
    #                 #         f"{activation}_{hidden_size}_{lr}"
    #                 #     )
    #                 #     explained_variances_unfrozen.append(explained_variance)

    # # create a dataset from main.dataset import Dataset, get_class_instance

    # # config = load_config("config.yaml")
    # # ruleset = config["rnn"]["train"]["ruleset"]
    # # all_rules = config["rnn"]["train"]["ruleset"] + config["rnn"]["pretrain"]["ruleset"]
    # # hp = {
    # #     "activation": "softplus",
    # #     "n_rnn": 128,
    # #     "learning_rate": 0.001,
    # #     "l2_h": 0.000001,
    # #     "l2_weight": 0.000001,
    # # }
    # # hp, log, optimizer = main.set_hyperparameters(
    # #     model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=ruleset
    # # )
    # # env = get_class_instance("PerceptualDecisionMakingT", config=hp)
    # # dataset = Dataset(env, batch_size=100, seq_len=100)
    # # inputs, labels = dataset.dataset()
    # # ipdb.set_trace()
    # # dataset.plot_env()
