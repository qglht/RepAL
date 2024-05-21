import main
from main import RNNLayer
from dsa_analysis import load_config, visualize
import torch
import pickle
import ipdb

config = load_config("config.yaml")
ruleset = config["rnn"]["pretrain"]["ruleset"]
activation = config["rnn"]["parameters"]["activations"][0]
hidden_size = config["rnn"]["parameters"]["n_rnn"][0]
lr = config["rnn"]["parameters"]["learning_rate"][0]
num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
devices = (
    [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    if num_gpus > 0
    else [torch.device("cpu")]
)
device = devices[0]

hp = {
    "activation": activation,
    "n_rnn": hidden_size,
    "learning_rate": lr,
    "l2_h": 0.000001,
    "l2_weight": 0.000001,
}
hp, log, optimizer = main.set_hyperparameters(
    model_dir="debug", hp=hp, ruleset=ruleset, rule_trains=ruleset
)

run_model = main.Run_Model(hp, RNNLayer, device)
main.train(run_model, optimizer, hp, log, freeze=False)
run_model.save(f"models/{activation}_{hidden_size}_{lr}_pretrain.pth")