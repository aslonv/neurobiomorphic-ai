"""
Advanced Neuroplasticity Layer with Latest Research

Implements cutting-edge neuroplasticity mechanisms based on recent neuroscience research:
- Multi-timescale synaptic plasticity
- Metaplasticity and homeostatic scaling
- Astrocyte-mediated plasticity modulation
- Sparse connectivity with dynamic pruning/sprouting
- Continual learning with synaptic consolidation
- Neural architecture search capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedNeuroplasticityLayer(nn.Module):
    """
    Advanced neuroplasticity layer implementing latest neuroscience research.
    
    Features:
    - Multi-timescale synaptic plasticity (LTP/LTD, homeostatic scaling)
    - Metaplasticity (plasticity of plasticity)
    - Astrocyte modulation of synaptic transmission
    - Dynamic structural plasticity (pruning/sprouting)
    - Continual learning with elastic weight consolidation
    """
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        sparsity: float = 0.9,
        enable_metaplasticity: bool = True,
        enable_structural_plasticity: bool = True,
        enable_continual_learning: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity = sparsity
        self.enable_metaplasticity = enable_metaplasticity
        self.enable_structural_plasticity = enable_structural_plasticity
        self.enable_continual_learning = enable_continual_learning
        
        # Core synaptic weights with sparse initialization
        self.weight = nn.Parameter(self._initialize_sparse_weights(output_size, input_size, sparsity))
        self.bias = nn.Parameter(torch.zeros(output_size))
        
        # Astrocyte modulation parameters
        self.astrocyte_activation = nn.Parameter(torch.ones(output_size))
        self.astrocyte_threshold = nn.Parameter(torch.ones(output_size) * 0.5)
        
        # Multi-compartment dendritic processing
        self.n_dendritic_segments = 10
        self.dendrite_segments = nn.Parameter(
            torch.randn(output_size, input_size, self.n_dendritic_segments) / math.sqrt(input_size)
        )
        self.dendritic_gates = nn.Parameter(torch.ones(output_size, self.n_dendritic_segments))
        
        # Synaptic plasticity mechanisms
        self.synaptic_tag = nn.Parameter(torch.zeros(output_size, input_size))
        self.protein_synthesis = nn.Parameter(torch.ones(output_size))
        self.consolidation_rate = nn.Parameter(torch.tensor(0.001))
        self.tag_decay = nn.Parameter(torch.tensor(0.99))
        self.protein_synthesis_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Metaplasticity parameters
        if enable_metaplasticity:
            self.metaplastic_state = nn.Parameter(torch.ones(output_size, input_size))
            self.metaplastic_decay = nn.Parameter(torch.tensor(0.995))
            self.metaplastic_scale = nn.Parameter(torch.tensor(0.1))
        
        # Homeostatic scaling parameters
        self.target_activity = nn.Parameter(torch.tensor(0.1))
        self.scaling_factor = nn.Parameter(torch.ones(output_size))
        self.activity_history = nn.Parameter(torch.zeros(output_size))
        self.history_decay = nn.Parameter(torch.tensor(0.99))
        
        # Structural plasticity parameters
        if enable_structural_plasticity:
            self.connection_strength = nn.Parameter(torch.ones(output_size, input_size))
            self.pruning_threshold = nn.Parameter(torch.tensor(0.01))
            self.sprouting_probability = nn.Parameter(torch.tensor(0.001))
            
        # Continual learning parameters (Elastic Weight Consolidation)
        if enable_continual_learning:
            self.fisher_information = nn.Parameter(torch.zeros(output_size, input_size), requires_grad=False)
            self.optimal_weights = nn.Parameter(torch.zeros(output_size, input_size), requires_grad=False)
            self.ewc_lambda = nn.Parameter(torch.tensor(1000.0))
            
        # Neuromodulator effects
        self.dopamine_level = nn.Parameter(torch.tensor(0.5))
        self.acetylcholine_level = nn.Parameter(torch.tensor(0.5))
        self.noradrenaline_level = nn.Parameter(torch.tensor(0.5))
        
        # Learning rate adaptation
        self.adaptive_lr = nn.Parameter(torch.ones(output_size, input_size) * 0.01)
        self.lr_adaptation_rate = nn.Parameter(torch.tensor(0.001))
        
    def _initialize_sparse_weights(self, out_size: int, in_size: int, sparsity: float) -> torch.Tensor:
        """Initialize sparse connectivity patterns."""
        weights = torch.randn(out_size, in_size) / math.sqrt(in_size)
        
        # Create sparse mask
        mask = torch.rand(out_size, in_size) > sparsity
        weights = weights * mask.float()
        
        return weights

    def forward(self, x: torch.Tensor, context: torch.Tensor, prev_activation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with advanced neuroplasticity mechanisms.
        
        Args:
            x: Input tensor [batch_size, input_size]
            context: Contextual information [batch_size, context_size] 
            prev_activation: Previous layer activation [batch_size, input_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Apply structural plasticity (dynamic connectivity)
        effective_weights = self.weight
        if self.enable_structural_plasticity:
            effective_weights = self._apply_structural_plasticity(effective_weights)
            
        # Astrocyte modulation based on context
        astro_modulation = self._compute_astrocyte_modulation(context)
        
        # Multi-compartment dendritic computation
        dendritic_output = self._dendritic_computation(x)
        
        # Synaptic transmission with astrocyte modulation
        # astro_modulation is [batch_size, output_size], effective_weights is [output_size, input_size]
        # Apply mean astrocyte modulation across batch for weight modulation
        mean_astro_mod = astro_modulation.mean(dim=0)  # [output_size]
        modulated_weights = effective_weights * mean_astro_mod.unsqueeze(-1)  # [output_size, input_size]
        synaptic_output = F.linear(x, modulated_weights, self.bias)
        
        # Combine somatic and dendritic contributions
        combined_output = synaptic_output + dendritic_output
        
        # Apply activation function
        output = F.relu(combined_output)
        
        # Update plasticity mechanisms during training
        if self.training:
            self._update_plasticity_mechanisms(x, output, prev_activation, context)
            
        return output
    
    def _compute_astrocyte_modulation(self, context: torch.Tensor) -> torch.Tensor:
        """Compute astrocyte-mediated synaptic modulation."""
        if context.dim() == 1:
            context = context.unsqueeze(0)
            
        # Simple context integration (can be made more sophisticated)
        context_signal = context.mean(dim=-1, keepdim=True)  # [batch_size, 1]
        
        # Expand to match output dimensions
        batch_size = context_signal.shape[0]
        context_expanded = context_signal.expand(batch_size, self.output_size)  # [batch_size, output_size]
        
        # Astrocyte activation based on context and current state
        activation_input = context_expanded * self.astrocyte_activation.unsqueeze(0)  # Broadcasting
        astro_mod = torch.sigmoid(activation_input)
        
        # Apply threshold for astrocyte activation
        threshold_mask = (astro_mod > self.astrocyte_threshold.unsqueeze(0)).float()
        astro_mod = astro_mod * threshold_mask
        
        return astro_mod
    
    def _dendritic_computation(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-compartment dendritic processing."""
        # Compute dendritic segment activations
        # x: [batch_size, input_size]
        # dendrite_segments: [output_size, input_size, n_segments]
        
        # Expand input for segment computation
        x_expanded = x.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, input_size, 1]
        segments_expanded = self.dendrite_segments.unsqueeze(0)  # [1, output_size, input_size, n_segments]
        
        # Compute segment activations (element-wise multiplication)
        segment_activations = torch.sum(x_expanded * segments_expanded, dim=2)  # [batch_size, output_size, n_segments]
        
        # Apply nonlinearity to each segment
        segment_outputs = F.relu(segment_activations)
        
        # Gate-controlled integration of dendritic segments
        gates = torch.sigmoid(self.dendritic_gates).unsqueeze(0)  # [1, output_size, n_segments]
        gated_segments = segment_outputs * gates
        
        # Integrate across segments
        dendritic_output = torch.sum(gated_segments, dim=-1)  # [batch_size, output_size]
        
        return dendritic_output
    
    def _apply_structural_plasticity(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply dynamic structural plasticity (pruning and sprouting)."""
        if not self.enable_structural_plasticity:
            return weights
            
        # Pruning: remove weak connections
        connection_mask = (torch.abs(weights) * self.connection_strength > self.pruning_threshold).float()
        
        # Sprouting: probabilistically add new connections
        if self.training:
            sprouting_mask = (torch.rand_like(weights) < self.sprouting_probability).float()
            # Only sprout where there are no existing connections
            sprouting_mask = sprouting_mask * (1 - connection_mask)
            
            # Add sprouted connections with small random weights
            sprouted_weights = sprouting_mask * torch.randn_like(weights) * 0.01
            
            # Update connection strengths
            new_connections = (connection_mask + sprouting_mask).clamp(0, 1)
            self.connection_strength.data = (
                self.connection_strength.data * 0.99 + new_connections * 0.01
            )
            
            return weights * connection_mask + sprouted_weights
        else:
            return weights * connection_mask
    
    def _update_plasticity_mechanisms(
        self, 
        x: torch.Tensor, 
        output: torch.Tensor, 
        prev_activation: torch.Tensor,
        context: torch.Tensor
    ) -> None:
        """Update various plasticity mechanisms."""
        
        # 1. Spike-timing-dependent plasticity (STDP)
        self._update_stdp(x, output, prev_activation)
        
        # 2. Homeostatic plasticity
        self._update_homeostatic_scaling(output)
        
        # 3. Metaplasticity
        if self.enable_metaplasticity:
            self._update_metaplasticity(x, output)
            
        # 4. Neuromodulator-dependent plasticity
        self._update_neuromodulator_effects(x, output, context)
        
        # 5. Synaptic consolidation
        self._update_synaptic_consolidation()
        
        # 6. Adaptive learning rates
        self._update_adaptive_learning_rates(x, output)
    
    def _update_stdp(self, x: torch.Tensor, output: torch.Tensor, prev_activation: torch.Tensor) -> None:
        """Update weights using spike-timing-dependent plasticity."""
        batch_size = x.shape[0]
        
        # Compute pre-post correlations
        pre_post = torch.bmm(output.unsqueeze(2), x.unsqueeze(1))  # [batch, output_size, input_size]
        
        # Timing-dependent component (simplified)
        # In real STDP, this would depend on precise spike timing
        if prev_activation is not None and prev_activation.shape == x.shape:
            timing_factor = torch.sigmoid(x - prev_activation.mean(dim=1, keepdim=True))
            timing_factor = timing_factor.unsqueeze(1)  # [batch, 1, input_size]
            pre_post = pre_post * timing_factor
        
        # Average across batch and compute STDP update
        stdp_update = pre_post.mean(dim=0) * self.adaptive_lr
        
        # Apply metaplasticity scaling if enabled
        if self.enable_metaplasticity:
            stdp_update = stdp_update * self.metaplastic_state
        
        # Update synaptic tags
        self.synaptic_tag.data = self.synaptic_tag.data * self.tag_decay + stdp_update
        
    def _update_homeostatic_scaling(self, output: torch.Tensor) -> None:
        """Update homeostatic scaling factors."""
        # Compute current activity levels
        current_activity = output.mean(dim=0)  # Average across batch
        
        # Update activity history with decay
        self.activity_history.data = (
            self.activity_history.data * self.history_decay + 
            current_activity * (1 - self.history_decay)
        )
        
        # Compute scaling factors to maintain target activity
        activity_error = self.target_activity - self.activity_history
        scaling_update = activity_error * 0.001  # Small learning rate for stability
        
        self.scaling_factor.data = torch.clamp(
            self.scaling_factor.data + scaling_update,
            0.1, 10.0  # Reasonable bounds for scaling
        )
        
    def _update_metaplasticity(self, x: torch.Tensor, output: torch.Tensor) -> None:
        """Update metaplastic state (plasticity of plasticity)."""
        # Compute activity-dependent metaplastic changes
        activity_level = output.mean(dim=0, keepdim=True)  # [1, output_size]
        input_activity = x.mean(dim=0, keepdim=True)  # [1, input_size]
        
        # Metaplastic update based on correlation of activities
        meta_update = torch.outer(activity_level.squeeze(), input_activity.squeeze())
        meta_update = meta_update * self.metaplastic_scale
        
        # Update metaplastic state with decay
        self.metaplastic_state.data = (
            self.metaplastic_state.data * self.metaplastic_decay +
            meta_update * (1 - self.metaplastic_decay)
        )
        
        # Keep metaplastic state in reasonable bounds
        self.metaplastic_state.data = torch.clamp(self.metaplastic_state.data, 0.1, 10.0)
    
    def _update_neuromodulator_effects(self, x: torch.Tensor, output: torch.Tensor, context: torch.Tensor) -> None:
        """Update neuromodulator levels and their effects on plasticity."""
        # Simplified neuromodulator dynamics based on context and activity
        
        # Dopamine: reward prediction error proxy
        reward_proxy = context.mean().item() if context.numel() > 0 else 0.5
        dopamine_update = (reward_proxy - self.dopamine_level.item()) * 0.01
        self.dopamine_level.data = torch.clamp(
            self.dopamine_level.data + dopamine_update, 0.0, 1.0
        )
        
        # Acetylcholine: attention/uncertainty proxy
        uncertainty_proxy = output.var().item()
        ach_update = uncertainty_proxy * 0.001 - self.acetylcholine_level.item() * 0.01
        self.acetylcholine_level.data = torch.clamp(
            self.acetylcholine_level.data + ach_update, 0.0, 1.0
        )
        
        # Noradrenaline: arousal/stress proxy
        arousal_proxy = output.mean().item()
        na_update = (arousal_proxy - 0.1) * 0.005 - self.noradrenaline_level.item() * 0.01
        self.noradrenaline_level.data = torch.clamp(
            self.noradrenaline_level.data + na_update, 0.0, 1.0
        )
        
        # Modulate plasticity based on neuromodulator levels
        modulation_factor = (
            self.dopamine_level * 1.0 +  # Dopamine enhances plasticity
            self.acetylcholine_level * 0.5 +  # ACh enhances attention-related plasticity
            self.noradrenaline_level * 0.3  # NA enhances arousal-dependent plasticity
        )
        
        # Apply modulation to consolidation rate
        self.consolidation_rate.data = torch.clamp(
            self.consolidation_rate.data * modulation_factor, 0.0001, 0.01
        )
    
    def _update_synaptic_consolidation(self) -> None:
        """Update synaptic consolidation based on tags and protein synthesis."""
        # Protein synthesis threshold gating
        protein_synthesis_active = (self.protein_synthesis > self.protein_synthesis_threshold).float()
        
        # Weight updates based on synaptic tags and protein synthesis
        weight_update = (
            self.consolidation_rate * 
            self.synaptic_tag * 
            protein_synthesis_active.unsqueeze(1) *
            self.scaling_factor.unsqueeze(1)  # Apply homeostatic scaling
        )
        
        # Apply continual learning constraints if enabled
        if self.enable_continual_learning:
            # Elastic Weight Consolidation penalty
            ewc_penalty = self.ewc_lambda * self.fisher_information * (self.weight - self.optimal_weights)
            weight_update = weight_update - 0.001 * ewc_penalty
        
        # Update weights
        self.weight.data = self.weight.data + weight_update
        
        # Decay protein synthesis
        self.protein_synthesis.data = torch.clamp(
            self.protein_synthesis.data * 0.99 + torch.randn_like(self.protein_synthesis) * 0.001,
            0.0, 2.0
        )
        
    def _update_adaptive_learning_rates(self, x: torch.Tensor, output: torch.Tensor) -> None:
        """Update adaptive learning rates based on local activity."""
        # Compute local activity measures
        input_activity = x.var(dim=0, keepdim=True)  # [1, input_size]
        output_activity = output.var(dim=0, keepdim=True)  # [1, output_size]
        
        # Compute activity-dependent learning rate modulation
        activity_product = torch.outer(output_activity.squeeze(), input_activity.squeeze())
        
        # Higher activity -> lower learning rate (to prevent runaway plasticity)
        lr_modulation = 1.0 / (1.0 + activity_product)
        
        # Update adaptive learning rates
        self.adaptive_lr.data = (
            self.adaptive_lr.data * (1 - self.lr_adaptation_rate) +
            lr_modulation * self.lr_adaptation_rate * 0.01
        )
        
        # Keep learning rates in reasonable bounds
        self.adaptive_lr.data = torch.clamp(self.adaptive_lr.data, 0.0001, 0.1)
    
    def consolidate_for_continual_learning(self, data_loader, criterion) -> None:
        """Consolidate weights for continual learning using Fisher Information."""
        if not self.enable_continual_learning:
            return
            
        self.eval()
        fisher_diagonal = torch.zeros_like(self.weight)
        
        for batch_x, batch_y in data_loader:
            # Forward pass
            output = self.forward(batch_x, batch_x, batch_x)  # Simplified context
            loss = criterion(output, batch_y)
            
            # Compute gradients
            loss.backward()
            
            # Accumulate Fisher Information (gradient^2)
            if self.weight.grad is not None:
                fisher_diagonal += self.weight.grad.data ** 2
                
        # Normalize by dataset size
        fisher_diagonal /= len(data_loader)
        
        # Store Fisher Information and optimal weights
        self.fisher_information.data = fisher_diagonal
        self.optimal_weights.data = self.weight.data.clone()
        
        logger.info("Consolidated weights for continual learning")
    
    def get_plasticity_stats(self) -> Dict[str, float]:
        """Get statistics about current plasticity state."""
        stats = {
            'mean_synaptic_tag': self.synaptic_tag.abs().mean().item(),
            'mean_protein_synthesis': self.protein_synthesis.mean().item(),
            'consolidation_rate': self.consolidation_rate.item(),
            'scaling_factor_mean': self.scaling_factor.mean().item(),
            'dopamine_level': self.dopamine_level.item(),
            'acetylcholine_level': self.acetylcholine_level.item(),
            'noradrenaline_level': self.noradrenaline_level.item(),
            'weight_sparsity': (self.weight.abs() < 1e-6).float().mean().item(),
        }
        
        if self.enable_metaplasticity:
            stats['metaplastic_state_mean'] = self.metaplastic_state.mean().item()
            
        if self.enable_structural_plasticity:
            stats['connection_strength_mean'] = self.connection_strength.mean().item()
            stats['active_connections'] = (self.connection_strength > self.pruning_threshold).float().mean().item()
            
        return stats
    
    def reset_plasticity_state(self) -> None:
        """Reset all plasticity-related parameters to initial state."""
        with torch.no_grad():
            self.synaptic_tag.zero_()
            self.protein_synthesis.fill_(1.0)
            self.activity_history.zero_()
            self.scaling_factor.fill_(1.0)
            
            if self.enable_metaplasticity:
                self.metaplastic_state.fill_(1.0)
                
            if self.enable_structural_plasticity:
                self.connection_strength.fill_(1.0)
                
            if self.enable_continual_learning:
                self.fisher_information.zero_()
                self.optimal_weights.zero_()
                
        logger.info("Reset all plasticity state parameters")
