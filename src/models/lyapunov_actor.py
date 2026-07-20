import numpy as np
import torch.nn as nn

from models.mlp import MLP


class LyapunovActor(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=(5, 5),
        action_dim=1,
        inner_activation=nn.Tanh,
        max_action=1.0,
    ):
        super(LyapunovActor, self).__init__()
        self.model = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            inner_activation=inner_activation,
            output_activation=nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, x):
        action = self.model(x)
        return self.max_action * action

    def forward_dreal(self, x_vars):
        out = self.model.forward_dreal(x_vars)[0]
        return np.array([self.max_action * out])
