""" Custom RNN implementation """

import torch
from torch import nn, jit
import math
import ipdb


class RNNCell_base(nn.Module):

    def __init__(
        self, input_size, hidden_size, nonlinearity, bias, init_type="kaiming"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.init_type = init_type  # Add initialization type

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_type == "kaiming":
            nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        elif self.init_type == "xavier":
            nn.init.xavier_uniform_(self.weight_ih)
            nn.init.xavier_uniform_(self.weight_hh)
        elif self.init_type == "orthogonal":
            nn.init.orthogonal_(self.weight_ih)
            nn.init.orthogonal_(self.weight_hh)
        elif self.init_type == "diag":
            # Initialize as diagonal matrices
            nn.init.constant_(self.weight_ih, 0)
            nn.init.constant_(self.weight_hh, 0)
            for i in range(min(self.hidden_size, self.input_size)):
                self.weight_ih[i, i] = 1.0  # Set diagonal elements to 1
            for i in range(self.hidden_size):
                self.weight_hh[i, i] = 1.0
        elif self.init_type == "randortho":
            # Initialize with random orthogonal matrices
            nn.init.orthogonal_(self.weight_ih)
            nn.init.orthogonal_(self.weight_hh)
        elif self.init_type == "randgauss":
            # Initialize with a random Gaussian distribution
            nn.init.normal_(self.weight_ih, mean=0, std=0.1)
            nn.init.normal_(self.weight_hh, mean=0, std=0.1)
        else:
            raise ValueError(f"Unknown initialization type: {self.init_type}")

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class RNNCell(RNNCell_base):  # Euler integration of rate-neuron network dynamics
    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity=None,
        decay=0.9,
        bias=True,
        init_type="kaiming",
    ):
        super().__init__(input_size, hidden_size, nonlinearity, bias, init_type)
        self.decay = decay  # torch.exp(- dt/tau)

    def forward(self, input, hidden):
        activity = self.nonlinearity(
            input @ self.weight_ih.t() + hidden @ self.weight_hh.t() + self.bias
        )
        hidden = self.decay * hidden + (1 - self.decay) * activity
        return hidden


class LeakyGRUCell(RNNCell_base):
    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity=None,
        decay=0.9,
        bias=True,
        init_type="kaiming",
    ):
        super().__init__(input_size, hidden_size, nonlinearity, bias, init_type)
        self.decay = decay

        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, input, hidden):
        gates = input @ self.weight_ih.t() + hidden @ self.weight_hh.t() + self.bias
        resetgate, updategate, newgate = gates.chunk(3, 1)

        resetgate = torch.sigmoid(resetgate)
        updategate = torch.sigmoid(updategate)
        newgate = self.nonlinearity(newgate)

        new_hidden = self.decay * hidden + (1 - self.decay) * newgate
        hidden = (1 - updategate) * hidden + updategate * new_hidden

        return hidden


class RNNLayer(nn.Module):
    def __init__(self, cell_type, *args, init_type="kaiming"):
        super().__init__()
        if cell_type == "leaky_rnn":
            self.rnncell = RNNCell(*args, init_type=init_type)
        elif cell_type == "leaky_gru":
            self.rnncell = LeakyGRUCell(*args, init_type=init_type)
        else:
            raise ValueError("Unsupported cell type: {}".format(cell_type))

    def forward(self, input, hidden_init):
        inputs = input.unbind(0)  # inputs has dimension [Time, batch, n_input]
        hidden = hidden_init[0]  # initial state has dimension [1, batch, n_input]
        outputs = []
        for i in range(len(inputs)):  # looping over the time dimension
            hidden = self.rnncell(inputs[i], hidden)
            outputs += [hidden]  # vanilla RNN directly outputs the hidden state
        return torch.stack(outputs), hidden
