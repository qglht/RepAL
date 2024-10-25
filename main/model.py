""" Network model that works with train_pytorch.py """

import torch
from torch import nn, jit
import numpy as np
import copy
from mambapy.mamba_lm import MambaLM, MambaLMConfig
from mambapy.mamba import ResidualBlock, MambaConfig, RMSNorm
import ipdb


class Model(nn.Module):
    """
    Model class for the RNN model

    Args:
    hp (dict): The hyperparameters
    RNNLayer (nn.Module): The RNN layer

    Attributes:
    n_rnn (int): The number of RNN units
    rnn (nn.Module): The RNN layer
    readout (nn.Linear): The readout layer
    """

    def __init__(self, hp, RNNLayer):
        super().__init__()
        n_input, n_rnn, n_output, decay, init_type = (
            hp["n_input"],
            hp["n_rnn"],
            hp["n_output"],
            hp["decay"],
            hp["init_type"],
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
        self.rnn = RNNLayer(
            hp["rnn_type"], n_input, n_rnn, nonlinearity, decay, init_type
        )
        self.readout = nn.Linear(n_rnn, n_output, bias=False)

    def forward(self, x):
        hidden0 = torch.zeros(
            [1, x.shape[1], self.n_rnn], device=x.device
        )  # initial hidden state
        hidden, _ = self.rnn(x, hidden0)
        output = self.readout(hidden)
        return output, hidden


class Run_Model(nn.Module):  # (jit.ScriptModule):
    """
    Model class for the RNN model

    Args:
    hp (dict): The hyperparameters
    RNNLayer (nn.Module): The RNN layer
    device (torch.device): The device to run the model on

    Attributes:
    hp (dict): The hyperparameters
    model (Model): The RNN model
    device (torch.device): The device to run the model on
    loss_fnc (nn.Module): The loss function
    """

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
        # loss = (loss * mask).sum() / mask.sum()
        loss = loss.mean()
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


class Mamba(nn.Module):
    """
    Mamba class for the Mamba model

    Args:
    config (MambaConfig): The configuration for the Mamba model

    Attributes:
    config (MambaConfig): The configuration for the Mamba model
    layers (nn.ModuleList): The list of residual blocks

    Methods:
    forward: Forward pass through the
    model
    step: Forward pass for one time step

    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, x, stop_at_layer=None):
        # x : (B, L, D)

        # y : (B, L, D)

        for i, layer in enumerate(self.layers):
            if stop_at_layer is not None:
                if i > stop_at_layer:
                    return x
                else:
                    x = layer(x)
            else:
                x = layer(x)

        return x

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class MambaSupervGym(MambaLM):
    """
    MambaSupervGym class for the Mamba model

    Args:
    hp (dict): The hyperparameters
    lm_config (MambaLMConfig): The configuration for the Mamba model
    device (torch.device): The device to run the model on

    Attributes:
    lm_config (MambaLMConfig): The configuration for the Mamba model
    device (torch.device): The device to run the model on
    config (MambaLMConfig): The configuration for the Mamba model
    hp (dict): The hyperparameters
    embedding (nn.Linear): The embedding layer
    mamba (Mamba): The Mamba model
    norm_f (RMSNorm): The normalization layer
    lm_head (nn.Linear): The linear layer
    loss_fnc (nn.Module): The loss function

    Methods:
    calculate_loss: Calculate the loss
    forward: Forward pass through the
    model
    forward_up_to: Forward pass up to a
    certain layer
    get_activations: Get the activations
    save: Save the model

    """

    def __init__(self, hp, lm_config, device):
        super().__init__(lm_config)
        self.lm_config = lm_config
        self.device = device
        self.config = lm_config.to_mamba_config()
        self.hp = hp

        # Initialize layers
        self.embedding = nn.Linear(
            self.hp["n_input"], self.config.d_model, bias=True
        ).to(self.device)
        self.mamba = Mamba(self.config).to(self.device)
        self.norm_f = RMSNorm(self.config.d_model).to(self.device)
        self.lm_head = nn.Linear(
            self.config.d_model, self.hp["n_output"], bias=False
        ).to(self.device)
        self.loss_fnc = nn.CrossEntropyLoss(reduction="none").to(self.device)

    def calculate_loss(self, output, mask, labels):
        loss = self.loss_fnc(output, labels)
        loss = loss.mean()
        loss_reg = 0
        for param in self.parameters():
            loss_reg += (
                param.abs().mean() * self.hp["l1_weight"]
                + param.norm() * self.hp["l2_weight"]
            )  #    Regularization cost  (L1 and L2 cost) on weights
        return loss, loss_reg

    def forward(self, tokens, labels, mask):
        """Function for all time steps directly

        Args:
            tokens (_type_): _description_
            labels (_type_): _description_
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        # tokens : (B, L)
        # logits : (B, L, vocab_size)
        x = self.embedding(tokens)

        x = self.mamba(x)
        x = self.norm_f(x)

        logits = self.lm_head(x)
        logits = logits.view(-1, self.hp["n_output"])
        loss, loss_reg = self.calculate_loss(logits, mask, labels)

        return (
            loss,
            loss_reg,
            logits,
            None,
            labels,
        )

    def forward_up_to(self, tokens, layer):
        # tokens : (B, L)
        # layer (1->n_layers): will stop the forward pass just after this layer

        # x : (B, L, D) activations after {layer}

        x = self.embedding(tokens)
        x = self.mamba(x, stop_at_layer=layer)

        return x

    def get_activations(self, token):
        """Function for one time step at a time: to do it for all time steps, loop for first dimension

        Args:
            token (_type_): _description_
            caches (_type_): _description_

        Returns:
            _type_: _description_
        """
        # # token : (B)
        # # caches : [cache(layer) for all layers], cache : (h, inputs)
        # # logits : (B, vocab_size)
        # # caches : [cache(layer) for all layers], cache : (h, inputs)

        # # loop over first dimension of token
        # # create cache for each layer
        cache_init = [
            (
                None,
                torch.zeros(
                    (
                        token.shape[0],
                        int(self.lm_config.d_model * self.lm_config.expand_factor),
                        self.hp["n_input"],
                    )
                ).to(self.device),
            )
            for _ in range(self.lm_config.n_layers)
        ]
        caches_list = []
        caches = cache_init
        for i in range(token.shape[1]):
            x = self.embedding(token[:, i, :])
            x, caches = self.mamba.step(x, caches)
            caches_list.append(copy.deepcopy(caches))

        caches_hidden = torch.stack(
            [caches_list[i][0][0] for i in range(len(caches_list))], dim=0
        )
        hidden = caches_hidden.reshape(
            caches_hidden.shape[0],
            caches_hidden.shape[1],
            caches_hidden.shape[2] * caches_hidden.shape[3],
        )
        return hidden

    def save(self, path):
        torch.save(self.state_dict(), path)


def load_model(path, hp, RNNLayer, device):
    model = Run_Model(hp, RNNLayer, device)
    with open(path, "rb") as f:
        state_dict = torch.load(f, map_location=device)
    model.model.load_state_dict(state_dict)
    return model


def load_model_mamba(path, hp, lm_config, device):
    model = MambaSupervGym(hp, lm_config, device)
    with open(path, "rb") as f:
        state_dict = torch.load(f, map_location=device)
    model.load_state_dict(state_dict)
    return model
