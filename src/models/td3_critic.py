import torch
import torch.nn as nn

from models.mlp import MLP


class TD3Critic(nn.Module):
    def __init__(self, state_dim, hidden_sizes, action_dim):
        super(TD3Critic, self).__init__()
        input_size = state_dim + action_dim
        self.Q1 = MLP(input_size, hidden_sizes, 1)
        self.Q2 = MLP(input_size, hidden_sizes, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.Q1(sa)
        q2 = self.Q2(sa)
        return q1, q2

    def Q1_value(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.Q1(sa)
