""" Network model that works with train_pytorch.py """

import torch
from torch import nn, jit
from main.dataset import Dataset, get_class_instance
import numpy as np
import ipdb
# from main import *


def _gen_feed_dict(inputs, labels, rule, hp, device):
    # Ensure all data is already on the correct device
    inputs = torch.as_tensor(inputs, device=device)
    labels = torch.as_tensor(labels, device=device)
    n_time, batch_size = inputs.shape[:2]

    new_shape = [n_time, batch_size, hp["rule_start"] + hp["n_rule"]]
    x = torch.zeros(new_shape, dtype=torch.float32, device=device)
    ind_rule = hp["rules"].index(rule)
    x[:, :, :hp["rule_start"]] = inputs
    x[:, :, hp["rule_start"] + ind_rule] = 1
    inputs = x
    return inputs, labels

class Model(nn.Module):
    def __init__(self, hp, RNNLayer):
        super().__init__()
        n_input, n_rnn, n_output, decay = (
            hp["n_input"],
            hp["n_rnn"],
            hp["n_output"],
            hp["decay"],
        )

        if (
            hp["activation"] == "relu"
        ):  # Type of activation runctions, relu, softplus, tanh, elu
            nonlinearity = nn.ReLU()
        elif hp["activation"] == "tanh":
            nonlinearity = nn.Tanh()
        elif hp["activation"] == "softplus":
            nonlinearity = nn.Softplus()
        else:
            raise NotImplementedError

        self.n_rnn = n_rnn
        self.rnn = RNNLayer(n_input, n_rnn, nonlinearity, decay)
        self.readout = nn.Linear(n_rnn, n_output, bias=False)

    def forward(self, x):
        hidden0 = torch.zeros(
            [1, x.shape[1], self.n_rnn], device=x.device
        )  # initial hidden state
        hidden, _ = self.rnn(x, hidden0)
        output = self.readout(hidden)
        return output, hidden


class Run_Model(nn.Module):  # (jit.ScriptModule):
    def __init__(self, hp, RNNLayer, device):
        super().__init__()
        self.hp = hp
        self.model = Model(hp, RNNLayer)
        self.device = device
        self.model.to(self.device)
        self.loss_fnc = (
            nn.MSELoss() if hp["loss_type"] == "lsq" else nn.CrossEntropyLoss()
        )

    def generate_trials(self, rule:str, hp, batch_size, seq_len):
        # return gen_trials(rule, hp, mode, batch_size, self.device)
        # TO DO : richer rule
        env = get_class_instance(rule, config=hp)
        return Dataset(env, batch_size, seq_len).dataset

    def calculate_loss(self, output, hidden, labels, hp):
        loss = self.loss_fnc(output, labels)
        loss_reg = (
            hidden.abs().mean() * hp["l1_h"] + hidden.norm() * hp["l2_h"]
        )  #    Regularization cost  (L1 and L2 cost) on hidden activity

        for param in self.parameters():
            loss_reg += (
                param.abs().mean() * hp["l1_weight"] + param.norm() * hp["l2_weight"]
            )  #    Regularization cost  (L1 and L2 cost) on weights
        return loss, loss_reg

    #     @jit.script_method
    def forward(self, rule, batch_size=None, seq_len=100):  # , **kwargs):
        hp = self.hp
        if batch_size is None:
            batch_size = hp["batch_size_test"]
        trial = self.generate_trials(rule, hp, batch_size, seq_len=seq_len)
        inputs, labels = trial()
        inputs, labels = _gen_feed_dict(inputs, labels, rule, hp, self.device)
        # why labels isn't it of good size???
        # check that they are on the good device
        output, hidden = self.model(inputs)
        output = output.view(-1, hp["n_output"])
        labels = labels.flatten()
        loss, loss_reg = self.calculate_loss(output, hidden, labels, hp)
        return (
            loss,
            loss_reg,
            output,
            hidden,
            labels,
        )

    def save(self, path):
        # Check if model is wrapped by DataParallel and save accordingly
        torch.save(self.model.state_dict(), path)


def load_model(path, hp, RNNLayer, device):
    model = Run_Model(hp, RNNLayer, device)
    state_dict = torch.load(path, map_location=device)
    model.model.load_state_dict(state_dict)
    return model
