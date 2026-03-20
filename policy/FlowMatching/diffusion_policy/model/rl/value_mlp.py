import torch
import torch.nn as nn


class CriticMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims: tuple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x):
        x = self.mlp1(x)
        return x

# # 正交初始化
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight, gain=2**0.5)
#         nn.init.constant_(m.bias, 0.0)
