"""
Advanced Neuroplastic Network with Continual Learning

Implements a neural network with advanced neuroplasticity mechanisms,
supporting continual learning, meta-learning, and adaptive architectures.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from .advanced_plasticity_layer import AdvancedNeuroplasticityLayer
import logging

logger = logging.getLogger(__name__)


class AdvancedNeuroplasticNetwork(nn.Module):
    """
    Neural network with advanced neuroplasticity mechanisms.
    
    Features:
    - Multi-layer neuroplastic processing
    - Adaptive network topology
    - Continual learning with consolidation
    - Meta-learning capabilities
    - Performance monitoring and adaptation
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        num_layers: int = 3,
        sparsity: float = 0.7,
        enable_metaplasticity: bool = True,
        enable_structural_plasticity: bool = True,
        enable_continual_learning: bool = True,
        adaptive_architecture: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.adaptive_architecture = adaptive_architecture
        self.enable_continual_learning = enable_continual_learning
        
        # Create neuroplastic layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size
                
            if i == num_layers - 1:
                layer_output_size = output_size
            else:
                layer_output_size = hidden_size
                
            layer = AdvancedNeuroplasticityLayer(
                input_size=layer_input_size,
                output_size=layer_output_size,
                sparsity=sparsity,
                enable_metaplasticity=enable_metaplasticity,
                enable_structural_plasticity=enable_structural_plasticity,
                enable_continual_learning=enable_continual_learning
            )
            
            self.layers.append(layer)
        
        # Adaptive architecture components
        if adaptive_architecture:
            self.layer_importance = nn.Parameter(torch.ones(num_layers))
            self.layer_gates = nn.Parameter(torch.ones(num_layers))
            
        # Global plasticity control
        self.global_plasticity_rate = nn.Parameter(torch.tensor(1.0))
        self.learning_phase = nn.Parameter(torch.tensor(0.0))  # 0: learning, 1: consolidation
        
        # Performance monitoring
        self.performance_history = []
        self.adaptation_threshold = 0.05
        
        # Meta-learning components
        self.task_embeddings = nn.Parameter(torch.randn(1, hidden_size))
        self.task_context_layer = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the neuroplastic network.
        
        Args:
            x: Input tensor [batch_size, input_size]
            context: Optional context information [batch_size, context_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Generate context if not provided
        if context is None:
            context = torch.zeros(batch_size, self.hidden_size, device=device)
        elif context.shape[-1] != self.hidden_size:
            context = self.task_context_layer(context)
            
        # Process task embedding
        task_context = self.task_embeddings.expand(batch_size, -1)
        combined_context = context + task_context
        
        # Forward pass through layers
        prev_activation = x
        current_activation = x
        
        for i, layer in enumerate(self.layers):
            # Apply layer gating if adaptive architecture is enabled
            if self.adaptive_architecture:
                gate_value = torch.sigmoid(self.layer_gates[i])
                layer_importance = self.layer_importance[i]
            else:
                gate_value = 1.0
                layer_importance = 1.0
                
            # Layer forward pass
            layer_output = layer(current_activation, combined_context, prev_activation)
            
            # Apply gating and importance scaling
            current_activation = layer_output * gate_value * layer_importance
            prev_activation = current_activation
            
        return current_activation
    
    def consolidate_learning(self, data_loader, criterion) -> None:
        """Consolidate learning across all layers for continual learning."""
        if not self.enable_continual_learning:
            return
            
        logger.info("Starting learning consolidation...")
        
        for i, layer in enumerate(self.layers):
            logger.info(f"Consolidating layer {i+1}/{len(self.layers)}")
            layer.consolidate_for_continual_learning(data_loader, criterion)
        
        # Update learning phase
        self.learning_phase.data = torch.tensor(1.0)  # Switch to consolidation mode
        
        logger.info("Learning consolidation completed")
    
    def adapt_architecture(self, performance_metric: float) -> None:
        """Adapt network architecture based on performance."""
        if not self.adaptive_architecture:
            return
            
        self.performance_history.append(performance_metric)
        
        # Only adapt if we have enough history
        if len(self.performance_history) < 10:
            return
            
        # Compute performance trend
        recent_perf = sum(self.performance_history[-5:]) / 5
        older_perf = sum(self.performance_history[-10:-5]) / 5
        performance_change = recent_perf - older_perf
        
        # Adapt based on performance trend
        if performance_change < -self.adaptation_threshold:
            # Performance is declining, increase plasticity
            self.global_plasticity_rate.data *= 1.1
            self._increase_layer_capacity()
            logger.info("Increased plasticity due to performance decline")
            
        elif performance_change > self.adaptation_threshold:
            # Performance is improving, can reduce plasticity slightly
            self.global_plasticity_rate.data *= 0.95
            self._optimize_layer_usage()
            logger.info("Optimized network due to good performance")
    
    def _increase_layer_capacity(self) -> None:
        """Increase capacity of underperforming layers."""
        with torch.no_grad():
            # Identify layers with low importance
            importances = torch.sigmoid(self.layer_importance)
            low_importance_mask = importances < 0.3
            
            # Increase gates for low-importance layers
            self.layer_gates.data[low_importance_mask] += 0.1
            self.layer_gates.data = torch.clamp(self.layer_gates.data, -2.0, 2.0)
    
    def _optimize_layer_usage(self) -> None:
        """Optimize layer usage by adjusting gates and importance."""
        with torch.no_grad():
            # Get current activations statistics
            importances = torch.sigmoid(self.layer_importance)
            gates = torch.sigmoid(self.layer_gates)
            
            # Reduce gates slightly for very high importance layers (prevent overfitting)
            high_importance_mask = importances > 0.8
            self.layer_gates.data[high_importance_mask] -= 0.05
            
            # Regularize towards uniform importance
            uniform_target = importances.mean()
            importance_deviation = importances - uniform_target
            self.layer_importance.data -= 0.01 * importance_deviation
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        stats = {
            'num_layers': len(self.layers),
            'global_plasticity_rate': self.global_plasticity_rate.item(),
            'learning_phase': self.learning_phase.item(),
            'performance_history_length': len(self.performance_history)
        }
        
        if self.adaptive_architecture:
            stats['layer_importance'] = torch.sigmoid(self.layer_importance).tolist()
            stats['layer_gates'] = torch.sigmoid(self.layer_gates).tolist()
            
        # Collect layer-specific statistics
        layer_stats = []
        for i, layer in enumerate(self.layers):
            layer_stat = layer.get_plasticity_stats()
            layer_stat['layer_index'] = i
            layer_stats.append(layer_stat)
            
        stats['layer_stats'] = layer_stats
        
        # Compute network-wide statistics
        all_weights = torch.cat([layer.weight.flatten() for layer in self.layers])
        stats['total_parameters'] = len(all_weights)
        stats['network_sparsity'] = (all_weights.abs() < 1e-6).float().mean().item()
        stats['weight_magnitude_mean'] = all_weights.abs().mean().item()
        stats['weight_magnitude_std'] = all_weights.std().item()
        
        return stats
    
    def reset_plasticity(self) -> None:
        """Reset plasticity state across all layers."""
        for layer in self.layers:
            layer.reset_plasticity_state()
            
        # Reset global parameters
        with torch.no_grad():
            self.global_plasticity_rate.fill_(1.0)
            self.learning_phase.fill_(0.0)
            
            if self.adaptive_architecture:
                self.layer_importance.fill_(1.0)
                self.layer_gates.fill_(1.0)
        
        # Reset performance history
        self.performance_history.clear()
        
        logger.info("Reset all plasticity state in network")
    
    def set_task_context(self, task_embedding: torch.Tensor) -> None:
        """Set task-specific context for meta-learning."""
        with torch.no_grad():
            if task_embedding.shape[-1] == self.hidden_size:
                self.task_embeddings.data = task_embedding.unsqueeze(0)
            else:
                logger.warning(f"Task embedding size {task_embedding.shape[-1]} does not match hidden size {self.hidden_size}")
    
    def get_effective_capacity(self) -> float:
        """Compute the effective capacity of the network."""
        if not self.adaptive_architecture:
            return 1.0
            
        # Compute effective capacity based on layer gates and importance
        gates = torch.sigmoid(self.layer_gates)
        importance = torch.sigmoid(self.layer_importance)
        
        layer_capacities = gates * importance
        effective_capacity = layer_capacities.mean().item()
        
        return effective_capacity
    
    def prune_network(self, pruning_threshold: float = 0.01) -> int:
        """
        Prune network by removing small weights across all layers.
        
        Args:
            pruning_threshold: Threshold below which weights are set to zero
            
        Returns:
            Number of weights pruned
        """
        total_pruned = 0
        
        with torch.no_grad():
            for layer in self.layers:
                weight_mask = layer.weight.abs() < pruning_threshold
                pruned_count = weight_mask.sum().item()
                total_pruned += pruned_count
                
                layer.weight.data[weight_mask] = 0.0
                
        logger.info(f"Pruned {total_pruned} weights below threshold {pruning_threshold}")
        return total_pruned
    
    def grow_network(self, growth_factor: float = 0.1) -> None:
        """
        Grow network capacity by adding new connections.
        
        Args:
            growth_factor: Fraction of new connections to add
        """
        if not hasattr(self.layers[0], 'enable_structural_plasticity') or not self.layers[0].enable_structural_plasticity:
            logger.warning("Structural plasticity not enabled, cannot grow network")
            return
            
        with torch.no_grad():
            for layer in self.layers:
                # Find zero weights that can be activated
                zero_mask = (layer.weight.abs() < 1e-8)
                n_zeros = zero_mask.sum().item()
                
                if n_zeros > 0:
                    n_grow = min(int(n_zeros * growth_factor), n_zeros)
                    zero_indices = torch.nonzero(zero_mask, as_tuple=False)
                    
                    # Randomly select indices to grow
                    if n_grow > 0:
                        perm = torch.randperm(len(zero_indices))[:n_grow]
                        grow_indices = zero_indices[perm]
                        
                        # Initialize new weights with small random values
                        for idx in grow_indices:
                            i, j = idx[0].item(), idx[1].item()
                            layer.weight.data[i, j] = torch.randn(1).item() * 0.01
                            
        logger.info(f"Grew network with growth factor {growth_factor}")
        
    def save_plasticity_state(self, filepath: str) -> None:
        """Save the current plasticity state."""
        plasticity_state = {
            'layer_states': [layer.state_dict() for layer in self.layers],
            'global_plasticity_rate': self.global_plasticity_rate.item(),
            'learning_phase': self.learning_phase.item(),
            'performance_history': self.performance_history,
            'task_embeddings': self.task_embeddings.data.clone()
        }
        
        if self.adaptive_architecture:
            plasticity_state['layer_importance'] = self.layer_importance.data.clone()
            plasticity_state['layer_gates'] = self.layer_gates.data.clone()
            
        torch.save(plasticity_state, filepath)
        logger.info(f"Saved plasticity state to {filepath}")
        
    def load_plasticity_state(self, filepath: str) -> None:
        """Load plasticity state from file."""
        plasticity_state = torch.load(filepath, map_location=next(self.parameters()).device)
        
        # Load layer states
        for layer, state in zip(self.layers, plasticity_state['layer_states']):
            layer.load_state_dict(state)
            
        # Load global parameters
        self.global_plasticity_rate.data = torch.tensor(plasticity_state['global_plasticity_rate'])
        self.learning_phase.data = torch.tensor(plasticity_state['learning_phase'])
        self.performance_history = plasticity_state['performance_history']
        self.task_embeddings.data = plasticity_state['task_embeddings']
        
        if self.adaptive_architecture and 'layer_importance' in plasticity_state:
            self.layer_importance.data = plasticity_state['layer_importance']
            self.layer_gates.data = plasticity_state['layer_gates']
            
        logger.info(f"Loaded plasticity state from {filepath}")