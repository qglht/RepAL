""" Network model that works with train_pytorch.py """

import torch
from torch import nn, jit
import numpy as np
from mambapy.mamba_lm import MambaLM, MambaLMConfig
from mambapy.mamba import Mamba, MambaConfig, RMSNorm


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
        elif hp["activation"] == "leaky_relu":
            nonlinearity = nn.LeakyReLU()
        elif hp["activation"] == "tanh":
            nonlinearity = nn.Tanh()
        elif hp["activation"] == "softplus":
            nonlinearity = nn.Softplus()
        else:
            raise NotImplementedError

        self.n_rnn = n_rnn
        self.rnn = RNNLayer(hp["rnn_type"], n_input, n_rnn, nonlinearity, decay)
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
            nn.MSELoss()
            if hp["loss_type"] == "lsq"
            else nn.CrossEntropyLoss(reduction="none")
        )

    def calculate_loss(self, output, mask, labels, hidden, hp):
        # use mask to calculate loss of crossentropyloss
        loss = self.loss_fnc(output, labels)
        loss = (loss * mask).mean()
        loss_reg = (
            hidden.abs().mean() * hp["l1_h"] + hidden.norm() * hp["l2_h"]
        )  #    Regularization cost  (L1 and L2 cost) on hidden activity

        for param in self.parameters():
            loss_reg += (
                param.abs().mean() * hp["l1_weight"] + param.norm() * hp["l2_weight"]
            )  #    Regularization cost  (L1 and L2 cost) on weights
        return loss, loss_reg

    #     @jit.script_method
    def forward(self, inputs, labels, mask):  # , **kwargs):
        hp = self.hp
        output, hidden = self.model(inputs)
        output = output.view(-1, hp["n_output"])
        loss, loss_reg = self.calculate_loss(output, mask, labels, hidden, hp)
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


class MambaSupervGym(MambaLM):
    def __init__(self, num_actions, num_obs, lm_config):
        super().__init__(lm_config)
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()

        # self.embedding = nn.Embedding(num_obs, self.config.d_model)
        self.embedding = nn.Linear(num_obs, self.config.d_model, bias=True)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, num_actions, bias=False)
        # self.lm_head.weight = self.embedding.weight

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x = self.mamba(x)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits

    def step(self, token, caches):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        x, caches = self.mamba.step(x, caches)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits, caches


def load_model(path, hp, RNNLayer, device):
    model = Run_Model(hp, RNNLayer, device)
    state_dict = torch.load(path, map_location=device)
    model.model.load_state_dict(state_dict)
    return model
