import numpy as np
import torch.nn as nn

from util.dreal import dreal_elementwise, dreal_sigmoid, require_dreal


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        inner_activation=nn.ReLU,
        output_activation=None,
    ):
        """
        :param inner_activation: activation function to use for all hidden layers.
        """
        super(MLP, self).__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(inner_activation())
            last_size = size
        layers.append(nn.Linear(last_size, output_size))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def forward_dreal(self, x_vars: np.ndarray):
        d = require_dreal()
        x = x_vars
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().cpu().numpy().astype(object)
                b = layer.bias.detach().cpu().numpy().astype(object)
                x = W @ x + b
            elif isinstance(layer, nn.Tanh):
                x = dreal_elementwise(x, d.tanh)
            elif isinstance(layer, nn.Sigmoid):
                x = dreal_elementwise(x, dreal_sigmoid)
            else:
                raise NotImplementedError(f"{layer} not supported in dReal mode")
        return x
