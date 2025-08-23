import torch
import torch.nn as nn
from .advanced_plasticity_layer import AdvancedNeuroplasticityLayer

class AdvancedNeuroplasticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([AdvancedNeuroplasticityLayer(
            input_size if i == 0 else hidden_size,
            hidden_size if i < num_layers - 1 else output_size
        ) for i in range(num_layers)])

    def forward(self, x, context):
        prev_activation = torch.zeros_like(x)
        for layer in self.layers:
            x = layer(x, context, prev_activation)
            prev_activation = x
        return x