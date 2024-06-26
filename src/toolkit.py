import warnings

import main
from main import RNNLayer
from dsa_analysis import load_config
import torch
import os
import pandas as pd
from dsa_analysis import simulation_line, simulation_lorenz, combine_simulations, load_config
import DSA
import pandas as pd
import numpy as np
from itertools import permutations
import similarity

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'

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
    all_rules = config['rules_analysis']
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

def normalize_within_unit_volume(tensor):
    # Ensure the input is a PyTorch tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")

    # Find the minimum and maximum values in the entire 3D tensor
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)

    # Calculate scaling factor to fit the entire tensor within the unit volume
    scale_factor = 1.0 / (max_value - min_value)

    # Normalize the tensor
    normalized_tensor = (tensor - min_value) * scale_factor
    # convert tensor to numpy array

    return normalized_tensor

def pipeline(group, rnn_type, activation, hidden_size, lr, batch_size, device):
    config = load_config("config.yaml")
    rules_pretrain = config['groups'][group]['pretrain']['ruleset']
    rules_train = config['groups'][group]['train']['ruleset']
    freeze = config['groups'][group]['train']['frozen']
    all_rules = config['all_rules']
    hp = {
        "rnn_type": rnn_type,
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.0001,
        "l2_weight": 0.0001,
        "num_epochs": 50,
        "batch_size_train":batch_size
    }
    model_name = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}"
    path_pretrain_folder = os.path.join(f"models/{group}", model_name + f"_pretrain")
    path_pretrain_model = os.path.join(f"models/{group}", model_name + f"_pretrain.pth")
    path_train_folder = os.path.join(f"models/{group}", model_name + f"_train")
    path_train_model = os.path.join(f"models/{group}", model_name + f"_train.pth")

    # Pretraining
    print(f"Pretraining model {model_name} for group {group}")
    if not os.path.exists(path_pretrain_model):
        if rules_pretrain:
            hp, log, optimizer = main.set_hyperparameters(
            model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_pretrain
        )
            run_model = main.Run_Model(hp, RNNLayer, device)
            main.train(run_model, optimizer, hp, log, path_pretrain_folder)
            run_model.save(path_pretrain_model)

    # Training
    print(f"Training model {model_name} for group {group}")
    if not os.path.exists(path_train_model):
        if rules_train:
            if rules_pretrain:
                # load the model first
                hp, log, optimizer = main.set_hyperparameters(
                model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_train)
                run_model = main.load_model(
                        path_pretrain_model,
                        hp,
                        RNNLayer,
                        device=device,
                    )
                main.train(run_model, optimizer, hp, log, path_train_folder, freeze=freeze)
                run_model.save(path_train_model)
            else:
                hp, log, optimizer = main.set_hyperparameters(
                    model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_train
                )
                run_model = main.Run_Model(hp, RNNLayer, device)
                main.train(run_model, optimizer, hp, log, path_train_folder, freeze=freeze)
                run_model.save(path_train_model)
        else:
            # if rules_train is empty, then we don't train the model
            hp, log, optimizer = main.set_hyperparameters(
                    model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_train
                )
            run_model = main.Run_Model(hp, RNNLayer, device)
            run_model.save(path_train_model)
    return

def generate_data(env):
    config = load_config("config.yaml")
    all_rules = config["all_rules"]
    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp={}, ruleset=all_rules
    )
    main.generate_data(env, hp, mode="train", num_pregenerated=10000)

    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp={"mode": "test"}, ruleset=all_rules
    )
    main.generate_data(env, hp, mode="test", num_pregenerated=1000)

def get_dynamics_model(rnn_type, activation, hidden_size, lr, model, group,device, n_components=3):
    # Load configuration and set hyperparameters
    config = load_config("config.yaml")
    ruleset = config["rules_analysis"][-1]
    all_rules = config["rules_analysis"]

    hp = {
        "rnn_type": rnn_type,
        "activation": activation,
        "n_rnn": hidden_size,
        "learning_rate": lr,
        "l2_h": 0.00001,
        "l2_weight": 0.00001,
        "mode": "test",
    }
    hp, _, _ = main.set_hyperparameters(
        model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=ruleset
    )
    run_model = main.load_model(
        f"models/{group}/{model}",
        hp,
        RNNLayer,
        device=device,
    )
    h = main.representation(run_model, all_rules)
    h_trans, explained_variance = main.compute_pca(h, n_components=n_components)
    tensor_on_cpu = h_trans[("AntiPerceptualDecisionMakingDelayResponseT", "stimulus")].cpu()
    return tensor_on_cpu.detach().numpy(), explained_variance

def dissimilarity_over_learning(group1, group2, rnn_type, activation, hidden_size, lr, batch_size, device):
    config = load_config("config.yaml")
    all_rules = config["rules_analysis"]

    # paths for checkpoints
    model_name = f"{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}"
    path_train_folder1 = os.path.join(f"models/{group1}", model_name + f"_train")
    path_train_folder2 = os.path.join(f"models/{group2}", model_name + f"_train")

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
        if len(checkpoint_files_1) < len(checkpoint_files_2):
            index_epochs = corresponding_training_time(len(checkpoint_files_1), len(checkpoint_files_2))
            for epoch in index_epochs:
                checkpoint1 = torch.load(os.path.join(path_train_folder1, checkpoint_files_1[index_epochs.index(epoch)]), map_location=device)
                run_model1.load_state_dict(checkpoint1['model_state_dict'])
                checkpoint2 = torch.load(os.path.join(path_train_folder2, checkpoint_files_2[epoch]), map_location=device)
                run_model2.load_state_dict(checkpoint2['model_state_dict'])
                models_to_compare.extend([(run_model1, run_model2)])
        else:
            index_epochs = corresponding_training_time(len(checkpoint_files_2), len(checkpoint_files_1))
            for epoch in index_epochs:
                checkpoint1 = torch.load(os.path.join(path_train_folder1, checkpoint_files_1[epoch]), map_location=device)
                run_model1.load_state_dict(checkpoint1['model_state_dict'])
                checkpoint2 = torch.load(os.path.join(path_train_folder2, checkpoint_files_2[index_epochs.index(epoch)]), map_location=device)
                run_model2.load_state_dict(checkpoint2['model_state_dict'])
                models_to_compare.extend([(run_model1, run_model2)])

        # compute the curves for models and dissimilarities
    curves = [(get_curves(tuple_model[0], all_rules, components=15), get_curves(tuple_model[1], all_rules, components=15)) for tuple_model in models_to_compare]
    for epoch_index in range(len(index_epochs)):
        dissimilarities_over_learning["cka"].append(1-cka_measure(curves[epoch_index][0], curves[epoch_index][1]))
        dissimilarities_over_learning["procrustes"].append(1-procrustes_measure(curves[epoch_index][0], curves[epoch_index][1]))
        dsa_comp = DSA.DSA(
            curves[epoch_index][0], curves[epoch_index][1],
            n_delays=config["dsa"]["n_delays"],
            rank=config["dsa"]["rank"],
            delay_interval=config["dsa"]["delay_interval"],
            verbose=True,
            iters=1000,
            lr=1e-2,
            device=device
        )
        dissimilarities_over_learning["dsa"].append(dsa_comp.fit_score())
        return dissimilarities_over_learning        
    else:
        raise FileNotFoundError

def dsa_optimisation_compositionality(rank, n_delays, delay_interval, device, ordered=True, overwrite=True):
    path_file = f'data/dsa_results/{rank}_{n_delays}_{delay_interval}.csv' if not ordered else f'data/dsa_results/{rank}_{n_delays}_{delay_interval}_ordered.csv'
    if os.path.exists(path_file) and not overwrite:
        return
    else:
        config = load_config("config.yaml")
        # Define parameters
        dt = config['simulations']['dt']
        num_steps = config['simulations']['num_steps']
        num_samples = config['simulations']['num_samples']
        lorenz_parameters = config['simulations']['lorenz_parameters']

        # Run simulations line
        simulations_line = simulation_line(num_steps, num_samples)

        # Run simulations curve
        simulations_curve = simulation_lorenz(dt, lorenz_parameters['one_attractor'][1], num_samples, num_steps)

        # Run simulations Pattern1
        simulations_pattern1 = simulation_lorenz(dt, lorenz_parameters['two_stable_attractors'][0], num_samples, num_steps)

        # Run simulations Pattern2
        simulations_pattern2 = simulation_lorenz(dt, lorenz_parameters['two_stable_attractors'][2], num_samples, num_steps)

        # Run simulations line-curve-line-curve
        combined_simulations_line_curve_line = combine_simulations([simulations_line, simulations_curve, np.flip(simulations_line, axis=0), np.flip(simulations_curve, axis=0)], method='attach')

        motif_basis = [simulations_line, simulations_curve, simulations_pattern1,simulations_pattern2,combined_simulations_line_curve_line]
        motif_names = ['Line', 'Curve', 'Pattern1', 'Pattern2','Line-Curve-Line-Curve']
        motif_dict = {motif_names[i]: motif_basis[i] for i in range(len(motif_basis))}
        all_simulations_length_3 = list(permutations(motif_names, 3))
        all_simulations_combined = {permutation: combine_simulations([motif_dict[permutation[0]], motif_dict[permutation[1]], motif_dict[permutation[2]]],method='attach') for permutation in all_simulations_length_3}

        model = list(all_simulations_combined.values())
        model_names = list(all_simulations_combined.keys())

        dsa = DSA.DSA(model,n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=True,iters=1000,lr=1e-2, device=device)
        similarities = dsa.fit_score()

        grouped_by_shared_elements = {i:[] for i in range(4)}
        for comp_motif_1 in model_names:
            for comp_motif_2 in model_names:
                if ordered:
                    grouped_by_shared_elements[same_order(comp_motif_1, comp_motif_2)].extend([(comp_motif_1, comp_motif_2)])
                else:
                    set_1 = set(comp_motif_1)
                    set_2 = set(comp_motif_2)
                    grouped_by_shared_elements[len(set_1.intersection(set_2))].extend([(comp_motif_1, comp_motif_2)])
    
        similarities_grouped_by_shared_elements = {i:[] for i in range(4)}
        for key in grouped_by_shared_elements:
            for tuple1, tuple2 in grouped_by_shared_elements[key]:
                similarities_grouped_by_shared_elements[key].append(similarities[model_names.index(tuple1), model_names.index(tuple2)])

        # # compute median of similarities for each group and plot similarity vs number of shared elements
        # median_similarities = {key: np.median(value) for key, value in similarities_grouped_by_shared_elements.items()}
        # std_devs = {key: np.std(value) for key, value in similarities_grouped_by_shared_elements.items()}

        # # Prepare data for plotting
        # keys = list(median_similarities.keys())
        # median_values = list(median_similarities.values())
        # std_dev_values = list(std_devs.values())

        # df = pd.DataFrame({'Number of shared elements': keys, 'Median similarity': median_values, 'Standard deviation': std_dev_values})
        # Prepare lists to store DataFrame rows
        data = []

        # Iterate over the shared elements
        for num_shared_elements, similarities in similarities_grouped_by_shared_elements.items():
            tuples = grouped_by_shared_elements[num_shared_elements]
            
            # Zip the similarities with the corresponding element pairs
            for (element1, element2), similarity in zip(tuples, similarities):
                # Sort the elements to ensure uniqueness
                sorted_pair = sorted([element1, element2])
                data.append([num_shared_elements, sorted_pair[0], sorted_pair[1], similarity])

        # Create DataFrame
        df = pd.DataFrame(data, columns=["number of shared elements", "element1", "element2", "similarity"])

        # Drop duplicates
        df = df.drop_duplicates(subset=["element1", "element2"])

        # check if the directory exists
        if not os.path.exists('data/dsa_results'):
            os.makedirs('data/dsa_results')
        df.to_csv(path_file)
        return 