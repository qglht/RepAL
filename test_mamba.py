from mambapy.mamba_lm import MambaLM, MambaLMConfig
from mambapy.mamba import Mamba, MambaConfig, RMSNorm
from dsa_analysis import (
    simulation_line,
    simulation_lorenz,
    combine_simulations,
    load_config,
)
import main
import ipdb

config = load_config("config.yaml")
hp = {
    "num_epochs": 50,
    "batch_size_train": 16,
    "learning_rate": 0.01,
}
all_rules = config["PDM"]["all_rules"]
rules_pretrain = config["PDM"]["groups"]["master"]["train"]["ruleset"]
hp, log, optimizer = main.set_hyperparameters(
    model_dir="debug", hp=hp, ruleset=all_rules, rule_trains=rules_pretrain
)
path_pretrain_folder = "debug"
path_pretrain_model = "debug/model.pt"
config = MambaLMConfig(
    d_model=16,
    n_layers=4,
    vocab_size=hp["n_input"],
    pad_vocab_size_multiple=1,  # https://github.com/alxndrTL/mamba.py/blob/main/mamba_lm.py#L27
    pscan=True,
)
run_model = main.MambaSupervGym(hp, config, device="cpu")
main.train(
    run_model,
    optimizer,
    hp,
    log,
    path_pretrain_folder,
    freeze=False,
    retrain=False,
    rnn=False,
)
run_model.save(path_pretrain_model)
