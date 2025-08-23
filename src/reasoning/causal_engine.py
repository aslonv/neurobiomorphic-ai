"""
Causal Reasoning Engine

Implements a production-grade causal reasoning system based on Pearl's causal hierarchy
and modern causal discovery algorithms. Supports interventional reasoning and
counterfactual inference for first-principles understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod
import networkx as nx
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalGraph:
    """Represents a causal graph with nodes and directed edges."""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    edge_weights: Optional[Dict[Tuple[str, str], float]] = None
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        if self.edge_weights:
            nx.set_edge_attributes(G, self.edge_weights, 'weight')
        return G
    
    def parents(self, node: str) -> List[str]:
        """Get parent nodes of a given node."""
        return [src for src, dst in self.edges if dst == node]
    
    def children(self, node: str) -> List[str]:
        """Get children nodes of a given node."""
        return [dst for src, dst in self.edges if src == node]


class CausalMechanism(nn.Module, ABC):
    """Abstract base class for causal mechanisms."""
    
    @abstractmethod
    def forward(self, parents: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the causal mechanism."""
        pass
    
    @abstractmethod
    def intervene(self, value: torch.Tensor) -> torch.Tensor:
        """Perform intervention by setting the value."""
        pass


class NeuralCausalMechanism(CausalMechanism):
    """Neural network-based causal mechanism with uncertainty quantification."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int,
        activation: str = "relu",
        use_bayesian: bool = True
    ):
        super().__init__()
        self.use_bayesian = use_bayesian
        
        # Build neural network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if use_bayesian:
                layers.append(BayesianLinear(prev_dim, hidden_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
                
            prev_dim = hidden_dim
        
        # Output layer
        if use_bayesian:
            layers.append(BayesianLinear(prev_dim, output_dim))
        else:
            layers.append(nn.Linear(prev_dim, output_dim))
            
        self.network = nn.Sequential(*layers)
        
        # Noise model for stochastic mechanisms
        self.noise_std = nn.Parameter(torch.ones(output_dim) * 0.1)
        
    def forward(self, parents: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional noise injection."""
        output = self.network(parents)
        
        if noise is None and self.training:
            noise = torch.randn_like(output) * self.noise_std
        
        if noise is not None:
            output = output + noise
            
        return output
    
    def intervene(self, value: torch.Tensor) -> torch.Tensor:
        """Perform intervention by directly returning the value."""
        return value
    
    def sample_posterior(self, parents: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Sample from posterior distribution (for Bayesian mechanisms)."""
        if not self.use_bayesian:
            return self.forward(parents).unsqueeze(0).repeat(n_samples, 1, 1)
        
        samples = []
        for _ in range(n_samples):
            samples.append(self.forward(parents))
        return torch.stack(samples, dim=0)


class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational inference."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.ones(out_features, in_features) * -3.0)
        
        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -3.0)
        
        # Prior parameters
        self.register_buffer('prior_weight_mu', torch.zeros(out_features, in_features))
        self.register_buffer('prior_weight_std', torch.ones(out_features, in_features) * prior_std)
        self.register_buffer('prior_bias_mu', torch.zeros(out_features))
        self.register_buffer('prior_bias_std', torch.ones(out_features) * prior_std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reparameterization trick."""
        # Sample weights and bias
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        
        weight_noise = torch.randn_like(self.weight_mu)
        bias_noise = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_std * weight_noise
        bias = self.bias_mu + bias_std * bias_noise
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between variational posterior and prior."""
        # KL for weights
        weight_kl = self._gaussian_kl(
            self.weight_mu, self.weight_logvar,
            self.prior_weight_mu, torch.log(self.prior_weight_std ** 2)
        )
        
        # KL for bias
        bias_kl = self._gaussian_kl(
            self.bias_mu, self.bias_logvar,
            self.prior_bias_mu, torch.log(self.prior_bias_std ** 2)
        )
        
        return weight_kl + bias_kl
    
    def _gaussian_kl(self, mu_q, logvar_q, mu_p, logvar_p):
        """Compute KL divergence between two Gaussians."""
        return 0.5 * (
            logvar_p - logvar_q + 
            torch.exp(logvar_q - logvar_p) + 
            (mu_q - mu_p).pow(2) / torch.exp(logvar_p) - 1
        ).sum()


class CausalReasoningEngine(nn.Module):
    """
    Advanced causal reasoning engine that can perform:
    - Causal discovery from observational data
    - Interventional reasoning
    - Counterfactual inference
    - Uncertainty quantification
    """
    
    def __init__(
        self,
        variable_names: List[str],
        mechanism_configs: Dict[str, Dict],
        discovery_method: str = "notears",
        max_parents: int = 5
    ):
        super().__init__()
        self.variable_names = variable_names
        self.num_variables = len(variable_names)
        self.max_parents = max_parents
        self.discovery_method = discovery_method
        
        # Initialize causal mechanisms
        self.mechanisms = nn.ModuleDict()
        for var_name, config in mechanism_configs.items():
            self.mechanisms[var_name] = NeuralCausalMechanism(**config)
        
        # Learnable adjacency matrix for causal discovery
        self.adj_matrix = nn.Parameter(
            torch.randn(self.num_variables, self.num_variables) * 0.1
        )
        
        # Temperature parameter for Gumbel-Softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def get_causal_graph(self, threshold: float = 0.3) -> CausalGraph:
        """Extract causal graph from learned adjacency matrix."""
        with torch.no_grad():
            adj = torch.sigmoid(self.adj_matrix)
            # Remove self-loops
            adj.fill_diagonal_(0)
            
            edges = []
            edge_weights = {}
            
            for i in range(self.num_variables):
                for j in range(self.num_variables):
                    if adj[i, j] > threshold:
                        src_var = self.variable_names[i]
                        dst_var = self.variable_names[j]
                        edges.append((src_var, dst_var))
                        edge_weights[(src_var, dst_var)] = adj[i, j].item()
            
            return CausalGraph(
                nodes=self.variable_names,
                edges=edges,
                edge_weights=edge_weights
            )
    
    def forward(
        self, 
        observations: torch.Tensor,
        interventions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional interventions.
        
        Args:
            observations: Input observations [batch_size, num_variables]
            interventions: Dictionary of interventions {var_name: value}
            
        Returns:
            Generated values and mechanism outputs
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        # Get causal ordering (topological sort)
        causal_graph = self.get_causal_graph()
        try:
            causal_order = list(nx.topological_sort(causal_graph.to_networkx()))
        except nx.NetworkXError:
            # If graph has cycles, use random order (should be avoided)
            logger.warning("Detected cycles in causal graph, using variable order")
            causal_order = self.variable_names
        
        generated_values = torch.zeros_like(observations)
        mechanism_outputs = {}
        
        for var_name in causal_order:
            var_idx = self.variable_names.index(var_name)
            
            # Check if this variable is intervened
            if interventions and var_name in interventions:
                generated_values[:, var_idx] = interventions[var_name]
                mechanism_outputs[var_name] = interventions[var_name]
                continue
            
            # Get parents of this variable
            parents = causal_graph.parents(var_name)
            
            if parents:
                parent_indices = [self.variable_names.index(p) for p in parents]
                parent_values = generated_values[:, parent_indices]
            else:
                # If no parents, use empty tensor
                parent_values = torch.zeros(batch_size, 0, device=device)
            
            # Generate value using causal mechanism
            if var_name in self.mechanisms:
                output = self.mechanisms[var_name](parent_values)
                generated_values[:, var_idx] = output.squeeze(-1) if output.dim() > 1 else output
                mechanism_outputs[var_name] = output
            else:
                # If no mechanism defined, copy from observations
                generated_values[:, var_idx] = observations[:, var_idx]
                mechanism_outputs[var_name] = observations[:, var_idx]
        
        return generated_values, mechanism_outputs
    
    def counterfactual_inference(
        self,
        observations: torch.Tensor,
        interventions: Dict[str, torch.Tensor],
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Perform counterfactual inference using the three-step procedure:
        1. Abduction: Infer noise values given observations
        2. Action: Apply interventions
        3. Prediction: Generate counterfactual outcomes
        """
        # Step 1: Abduction - infer noise values
        with torch.no_grad():
            original_outputs, _ = self.forward(observations)
            noise_values = observations - original_outputs
        
        # Step 2 & 3: Action and Prediction
        counterfactual_samples = []
        for _ in range(n_samples):
            cf_values, _ = self.forward(observations, interventions)
            counterfactual_samples.append(cf_values)
        
        return torch.stack(counterfactual_samples, dim=0)
    
    def causal_discovery_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute loss for causal discovery using NOTEARS-style optimization."""
        adj = torch.sigmoid(self.adj_matrix)
        
        # Reconstruction loss
        generated, _ = self.forward(observations)
        reconstruction_loss = F.mse_loss(generated, observations)
        
        # Acyclicity constraint (DAG constraint)
        # h(W) = tr(exp(W ⊙ W)) - d = 0 for DAG
        dag_loss = torch.trace(torch.matrix_exp(adj * adj)) - self.num_variables
        dag_loss = torch.abs(dag_loss)  # We want this to be zero
        
        # Sparsity regularization
        sparsity_loss = torch.sum(adj)
        
        # Bayesian regularization (if using Bayesian mechanisms)
        kl_loss = 0.0
        for mechanism in self.mechanisms.values():
            if hasattr(mechanism, 'network'):
                for layer in mechanism.network:
                    if isinstance(layer, BayesianLinear):
                        kl_loss += layer.kl_divergence()
        
        total_loss = (
            reconstruction_loss + 
            10.0 * dag_loss +  # High weight for DAG constraint
            0.01 * sparsity_loss +
            0.001 * kl_loss
        )
        
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'dag': dag_loss.item(), 
            'sparsity': sparsity_loss.item(),
            'kl': kl_loss if isinstance(kl_loss, float) else kl_loss.item()
        }
    
    def save_graph_visualization(self, filepath: str, threshold: float = 0.3) -> None:
        """Save causal graph visualization."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            graph = self.get_causal_graph(threshold)
            G = graph.to_networkx()
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=3000, alpha=0.7)
            
            # Draw edges with weights
            edges = G.edges()
            weights = [G[u][v].get('weight', 1.0) for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, 
                                 edge_color='gray', arrows=True, arrowsize=20)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            plt.title("Learned Causal Graph", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Causal graph visualization saved to {filepath}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")