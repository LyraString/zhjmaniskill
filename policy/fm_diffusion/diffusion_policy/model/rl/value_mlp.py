import torch
import torch.nn as nn
from typing import Iterable


class CriticMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims: Iterable[int],
        activation: str = "tanh",
        use_layer_norm: bool = False,
        orthogonal_init: bool = True,
        output_gain: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        hidden_dims = list(hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        act_cls = self._get_activation(activation)
        layers = []
        prev_dim = int(input_dim)
        for h in hidden_dims:
            h = int(h)
            layers.append(nn.Linear(prev_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp1 = nn.Sequential(*layers)

        if orthogonal_init:
            self._init_weights(activation=activation, output_gain=output_gain)

    def forward(self, x):
        x = self.mlp1(x.float())
        return x

    @staticmethod
    def _get_activation(name: str):
        name = name.lower()
        if name == "tanh":
            return nn.Tanh
        if name == "relu":
            return nn.ReLU
        if name == "silu":
            return nn.SiLU
        if name == "gelu":
            return nn.GELU
        raise ValueError(f"Unsupported activation: {name}")

    def _init_weights(self, activation: str, output_gain: float):
        if activation.lower() == "tanh":
            hidden_gain = nn.init.calculate_gain("tanh")
        elif activation.lower() == "relu":
            hidden_gain = nn.init.calculate_gain("relu")
        else:
            # SiLU/GELU 常用 ReLU 近似增益
            hidden_gain = nn.init.calculate_gain("relu")

        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linear_layers):
            gain = float(output_gain) if i == len(linear_layers) - 1 else hidden_gain
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)

