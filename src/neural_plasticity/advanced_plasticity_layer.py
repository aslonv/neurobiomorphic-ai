import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdvancedNeuroplasticityLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size) / math.sqrt(input_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.astrocyte_activation = nn.Parameter(torch.ones(output_size))
        self.dendrite_segments = nn.Parameter(torch.randn(output_size, input_size, 10) / math.sqrt(input_size))
        self.synaptic_tag = nn.Parameter(torch.zeros(output_size, input_size))
        self.protein_synthesis = nn.Parameter(torch.ones(output_size))
        self.consolidation_rate = 0.001
        self.tag_decay = 0.99
        self.protein_synthesis_threshold = 0.5

    def forward(self, x, context, prev_activation):
        # Simple astrocyte modulation using context as a scaling factor
        context_scale = torch.sigmoid(context.mean())  # Single scalar from context
        astro_mod = torch.sigmoid(self.astrocyte_activation * context_scale)
        
        # Dendritic computation - simplified to avoid shape issues
        dendritic_out = F.relu(torch.mm(x, self.dendrite_segments.mean(dim=2).T))
        
        # Synaptic transmission with astrocyte modulation
        modulated_weight = self.weight * astro_mod.unsqueeze(1)
        out = F.linear(x, modulated_weight, self.bias)
        
        # Simplified STDP - ensure correct dimensions
        if x.size(0) > 0 and out.size(0) > 0:
            # Create STDP matrix with same shape as synaptic_tag (output_size, input_size)
            x_avg = x.mean(dim=0)  # (input_size,)
            out_avg = out.mean(dim=0)  # (output_size,)
            stdp = torch.outer(out_avg, x_avg) * 0.01  # (output_size, input_size)
            
            # Synaptic tagging and capture
            self.synaptic_tag.data = self.synaptic_tag * self.tag_decay + stdp
        protein_synthesis = (self.protein_synthesis > self.protein_synthesis_threshold).float()
        
        # Weight update
        delta_w = self.consolidation_rate * self.synaptic_tag * protein_synthesis.unsqueeze(1)
        self.weight.data += delta_w
        
        # Homeostatic plasticity
        avg_act = out.mean(dim=0)
        target_act = 0.1
        homeostatic_factor = torch.clamp(target_act / (avg_act + 1e-6), 0.8, 1.2)
        self.weight.data *= homeostatic_factor.unsqueeze(1)
        
        return out + dendritic_out