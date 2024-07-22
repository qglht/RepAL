# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import neurogym as ngym
# import warnings
# from neurogym.utils import info, plotting
# import main

# warnings.filterwarnings("ignore")
# info.all_tasks()

# # Environment
# task = "PerceptualDecisionMaking-v0"
# kwargs = {"dt": 100}
# seq_len = 100
# data_batch_size = 16

# # Make supervised dataset
# dataset = ngym.Dataset(
#     task, env_kwargs=kwargs, batch_size=data_batch_size, seq_len=seq_len
# )
# inputs, labels = dataset()
# env = dataset.env
# ob_size = env.observation_space.shape[0]
# act_size = env.action_space.n
# print(type(dataset))

# rules_pretrain = config["PDM"]["groups"]["master"]["pretrain"]["ruleset"]
# rules_train = config["PDM"]["groups"]["master"]["train"]["ruleset"]
# freeze = config["PDM"]["groups"]["master"]["train"]["frozen"]
# all_rules = config["PDM"]["all_rules"]
# hp = {
#     "num_epochs": 50,
#     "batch_size_train": batch_size,
#     "learning_rate": learning_rate,
# }
# hp, log, optimizer = main.set_hyperparameters(
#     model_dir="debug",
#     hp=hp,
#     ruleset=all_rules,
#     rule_trains=rules_pretrain,
# )
# config = MambaLMConfig(
#     d_model=16,
#     n_layers=1,
#     vocab_size=hp["n_input"],
#     pad_vocab_size_multiple=1,  # https://github.com/alxndrTL/mamba.py/blob/main/mamba_lm.py#L27
#     pscan=True,
# )
# run_model = main.MambaSupervGym(hp, config, device="cpu")
# optim = torch.optim.Adam(model.parameters(), lr=1e-2)

# for i in range(2000):
#     inputs, labels = dataset()
#     inputs = np.swapaxes(inputs, 0, 1)
#     labels = np.swapaxes(labels, 0, 1)
#     obs = torch.from_numpy(inputs).type(torch.float).to(device)
#     act = torch.from_numpy(labels).type(torch.long).to(device)

#     logits = model(obs)

#     if i == 0:
#         pre_train_logits = logits
#         acc = accuracy(pre_train_logits, act)
#         print(f"Pre Train Accuracy: {acc * 100:.2f}%")

#     # Reshape logits to shape [(batch * images), classes]
#     logits_flat = logits.view(-1, act_size)

#     # Reshape true class indices to shape [(batch * images)]
#     true_class_indices_flat = act.flatten()

#     # Calculate cross entropy loss
#     loss = F.cross_entropy(logits_flat, true_class_indices_flat)

#     optim.zero_grad()
#     loss.backward()
#     optim.step()

#     if i % 200 == 0:
#         print(loss.item())
#         acc = accuracy(logits, act)
#         print(f"Train Accuracy: {acc * 100:.2f}%")
