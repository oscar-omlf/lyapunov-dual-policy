import torch.nn as nn

from models.mlp import MLP


class TD3Actor(nn.Module):
    def __init__(self, input_size, hidden_sizes, action_dim, max_action):
        super(TD3Actor, self).__init__()
        self.max_action = max_action
        self.model = MLP(
            input_size, hidden_sizes, action_dim, output_activation=nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.model(state)
