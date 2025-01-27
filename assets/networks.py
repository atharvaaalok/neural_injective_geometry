import torch
from torch import nn
import numpy as np

from .SmoothMonotonicNN import SmoothMonotonicNN

class InjectiveNet(nn.Module):
    def __init__(self, layer_count, act_fn):
        super().__init__()

        # Define the transformation from t on the [0, 1] interval to unit circle for closed shapes
        self.closed_transform = lambda t: torch.hstack([
            torch.cos(2 * torch.pi * t),
            torch.sin(2 * torch.pi * t)
        ])

        layers = []
        for i in range(layer_count):
            layers.append(nn.Linear(2, 2))
            layers.append(act_fn())
        
        self.linear_act_stack = nn.Sequential(*layers)
    
    def forward(self, t):
        x = self.closed_transform(t)
        x = self.linear_act_stack(x)
        return x



class NIG_Net(nn.Module):
    def __init__(self, layer_count_inj, smm_num_grps, smm_neurons_per_grp):
        super().__init__()

        # Transformation from t on the [0, 1] interval to unit circle for closed shapes
        self.closed_transform = lambda t: torch.hstack([
            torch.cos(2 * torch.pi * t),
            torch.sin(2 * torch.pi * t)
        ])

        self.layer_count_inj = layer_count_inj
        self.linear_layers = nn.ModuleList()
        self.monotonic_act = nn.ModuleList()

        for i in range(layer_count_inj):
            self.linear_layers.append(nn.Linear(2, 2))
            self.monotonic_act.append(SmoothMonotonicNN(
                n = 1, K = smm_num_grps, h_K = smm_neurons_per_grp, mask = np.array([1])
            ))
        
    def forward(self, t):
        x = self.closed_transform(t)

        for linear_layer, monotonic_act in zip(self.linear_layers, self.monotonic_act):
            # Apply linear transformation
            x = linear_layer(x)
            # Apply monotonic layers to each component of x separately
            x1, x2 = x[:, 0:1], x[:, 1:2]  # Efficient slicing
            x = torch.stack([monotonic_act(x1), monotonic_act(x2)], dim = -1)

        return x