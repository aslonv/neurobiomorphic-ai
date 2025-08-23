"""
Meta-Learning System for Few-Shot First-Principles Reasoning

Implements Model-Agnostic Meta-Learning (MAML) and Prototypical Networks
for rapid adaptation to new reasoning tasks with minimal examples.
Based on latest advances in meta-learning and few-shot learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class TaskBatch:
    """Represents a batch of few-shot learning tasks."""
    support_x: torch.Tensor  # [n_tasks, n_support, *input_shape]
    support_y: torch.Tensor  # [n_tasks, n_support, *output_shape]
    query_x: torch.Tensor    # [n_tasks, n_query, *input_shape]
    query_y: torch.Tensor    # [n_tasks, n_query, *output_shape]
    task_ids: Optional[List[str]] = None


class MetaLearningBase(nn.Module, ABC):
    """Base class for meta-learning algorithms."""
    
    @abstractmethod
    def meta_forward(self, task_batch: TaskBatch) -> Dict[str, torch.Tensor]:
        """Forward pass for meta-learning."""
        pass
    
    @abstractmethod
    def adapt_to_task(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor, 
        n_steps: int = 5
    ) -> nn.Module:
        """Adapt the model to a new task given support examples."""
        pass


class ReasoningNetwork(nn.Module):
    """Neural network for reasoning tasks with attention mechanisms."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism for reasoning
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=prev_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(prev_dim)
        
        # Output projection
        self.output_proj = nn.Linear(prev_dim, output_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional context for attention.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            context: Context tensor [batch_size, n_context, context_dim]
        
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Extract features
        features = self.feature_extractor(x)  # [batch_size, hidden_dim]
        
        # Apply attention if context is provided
        if self.use_attention and context is not None:
            # Reshape features for attention
            features_expanded = features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Apply multi-head attention
            attended, _ = self.attention(
                query=features_expanded,
                key=context,
                value=context
            )
            
            # Residual connection and normalization
            features = self.attention_norm(features + attended.squeeze(1))
        
        return self.output_proj(features)


class MAMLReasoningSystem(MetaLearningBase):
    """
    Model-Agnostic Meta-Learning for First-Principles Reasoning.
    
    Implements MAML algorithm adapted for reasoning tasks with
    enhanced gradient computation and stability improvements.
    """
    
    def __init__(
        self,
        base_network: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5,
        first_order: bool = False,
        allow_unused: bool = True
    ):
        super().__init__()
        self.base_network = base_network
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps
        self.first_order = first_order
        self.allow_unused = allow_unused
        
        # Meta-optimizer for outer loop
        self.meta_optimizer = torch.optim.Adam(
            self.base_network.parameters(), 
            lr=outer_lr
        )
        
        # Track adaptation statistics
        self.adaptation_stats = {
            'support_losses': [],
            'query_losses': [],
            'gradient_norms': []
        }
    
    def meta_forward(self, task_batch: TaskBatch) -> Dict[str, torch.Tensor]:
        """Perform meta-learning forward pass across multiple tasks."""
        n_tasks = task_batch.support_x.shape[0]
        device = task_batch.support_x.device
        
        meta_losses = []
        support_losses = []
        query_losses = []
        
        for task_idx in range(n_tasks):
            # Get task data
            support_x = task_batch.support_x[task_idx]
            support_y = task_batch.support_y[task_idx] 
            query_x = task_batch.query_x[task_idx]
            query_y = task_batch.query_y[task_idx]
            
            # Create task-specific model copy
            task_model = self._create_task_model()
            
            # Inner loop: adapt to support set
            support_loss, adapted_params = self._inner_loop_adaptation(
                task_model, support_x, support_y
            )
            
            # Evaluate on query set with adapted parameters
            query_pred = self._forward_with_params(query_x, adapted_params)
            query_loss = F.mse_loss(query_pred, query_y)
            
            meta_losses.append(query_loss)
            support_losses.append(support_loss)
            query_losses.append(query_loss.detach())
        
        # Average meta-loss across tasks
        meta_loss = torch.stack(meta_losses).mean()
        
        # Update statistics
        self.adaptation_stats['support_losses'].extend([l.item() for l in support_losses])
        self.adaptation_stats['query_losses'].extend([l.item() for l in query_losses])
        
        return {
            'meta_loss': meta_loss,
            'support_loss': torch.stack(support_losses).mean(),
            'query_loss': torch.stack(query_losses).mean()
        }
    
    def _create_task_model(self) -> nn.Module:
        """Create a copy of the base network for task adaptation."""
        return copy.deepcopy(self.base_network)
    
    def _inner_loop_adaptation(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """
        Perform inner loop adaptation using gradient descent.
        
        Returns:
            Tuple of (final_support_loss, adapted_parameters)
        """
        # Get initial parameters
        params = OrderedDict(model.named_parameters())
        
        for step in range(self.n_inner_steps):
            # Forward pass
            pred = model(support_x)
            loss = F.mse_loss(pred, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, 
                params.values(), 
                create_graph=not self.first_order,
                allow_unused=self.allow_unused
            )
            
            # Update parameters
            updated_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                if grad is not None:
                    updated_params[name] = param - self.inner_lr * grad
                else:
                    updated_params[name] = param
            
            params = updated_params
            
            # Update model parameters for next iteration
            self._set_model_params(model, params)
        
        # Final loss computation
        final_pred = model(support_x)
        final_loss = F.mse_loss(final_pred, support_y)
        
        return final_loss, params
    
    def _forward_with_params(
        self, 
        x: torch.Tensor, 
        params: OrderedDict
    ) -> torch.Tensor:
        """Forward pass using specific parameter values."""
        # Create temporary model with given parameters
        temp_model = copy.deepcopy(self.base_network)
        self._set_model_params(temp_model, params)
        return temp_model(x)
    
    def _set_model_params(self, model: nn.Module, params: OrderedDict):
        """Set model parameters from ordered dictionary."""
        for name, param in model.named_parameters():
            if name in params:
                param.data.copy_(params[name].data)
    
    def adapt_to_task(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_steps: int = 5
    ) -> nn.Module:
        """Adapt model to new task using support examples."""
        adapted_model = self._create_task_model()
        
        # Use inner loop adaptation
        _, adapted_params = self._inner_loop_adaptation(
            adapted_model, support_x, support_y
        )
        
        # Set adapted parameters
        self._set_model_params(adapted_model, adapted_params)
        
        return adapted_model
    
    def meta_train_step(self, task_batch: TaskBatch) -> Dict[str, float]:
        """Perform one meta-training step."""
        self.meta_optimizer.zero_grad()
        
        # Forward pass
        losses = self.meta_forward(task_batch)
        meta_loss = losses['meta_loss']
        
        # Backward pass
        meta_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.base_network.parameters(), 1.0)
        
        # Optimizer step
        self.meta_optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


class PrototypicalReasoningNetwork(MetaLearningBase):
    """
    Prototypical Networks adapted for reasoning tasks.
    
    Uses learned embeddings and prototype-based classification/regression
    for few-shot reasoning.
    """
    
    def __init__(
        self,
        embedding_network: nn.Module,
        distance_metric: str = "euclidean",
        temperature: float = 1.0
    ):
        super().__init__()
        self.embedding_network = embedding_network
        self.distance_metric = distance_metric
        self.temperature = temperature
        
    def meta_forward(self, task_batch: TaskBatch) -> Dict[str, torch.Tensor]:
        """Forward pass using prototypical networks."""
        n_tasks = task_batch.support_x.shape[0]
        
        meta_losses = []
        
        for task_idx in range(n_tasks):
            support_x = task_batch.support_x[task_idx]
            support_y = task_batch.support_y[task_idx]
            query_x = task_batch.query_x[task_idx] 
            query_y = task_batch.query_y[task_idx]
            
            # Embed support and query examples
            support_embeddings = self.embedding_network(support_x)
            query_embeddings = self.embedding_network(query_x)
            
            # Compute prototypes (centroids for each class/target)
            prototypes = self._compute_prototypes(support_embeddings, support_y)
            
            # Compute distances and predictions
            query_pred = self._predict_from_prototypes(query_embeddings, prototypes)
            
            # Compute loss
            task_loss = F.mse_loss(query_pred, query_y)
            meta_losses.append(task_loss)
        
        meta_loss = torch.stack(meta_losses).mean()
        
        return {'meta_loss': meta_loss}
    
    def _compute_prototypes(
        self, 
        embeddings: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute prototypes from embeddings and targets."""
        # For regression tasks, we need to discretize targets or use kernel methods
        # For simplicity, we'll compute a single prototype per unique target value
        
        unique_targets = torch.unique(targets, dim=0)
        prototypes = []
        
        for target in unique_targets:
            # Find examples with this target
            mask = torch.all(targets == target, dim=1)
            if mask.any():
                prototype = embeddings[mask].mean(dim=0)
                prototypes.append(prototype)
        
        if prototypes:
            return torch.stack(prototypes)
        else:
            # Fallback: return mean of all embeddings
            return embeddings.mean(dim=0).unsqueeze(0)
    
    def _predict_from_prototypes(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Make predictions based on distance to prototypes."""
        n_query = query_embeddings.shape[0]
        n_prototypes = prototypes.shape[0]
        
        # Compute distances
        if self.distance_metric == "euclidean":
            distances = torch.cdist(query_embeddings, prototypes)
        elif self.distance_metric == "cosine":
            # Cosine similarity (negative for distance)
            query_norm = F.normalize(query_embeddings, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            distances = -torch.mm(query_norm, proto_norm.t())
        
        # Convert distances to probabilities
        probs = F.softmax(-distances / self.temperature, dim=1)
        
        # For regression, we need to map back to target values
        # This is simplified - in practice you'd learn this mapping
        predictions = torch.sum(probs.unsqueeze(-1) * prototypes.unsqueeze(0), dim=1)
        
        return predictions
    
    def adapt_to_task(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_steps: int = 5
    ) -> nn.Module:
        """Adapt by computing prototypes from support set."""
        # For prototypical networks, adaptation is computing prototypes
        support_embeddings = self.embedding_network(support_x)
        prototypes = self._compute_prototypes(support_embeddings, support_y)
        
        # Return a wrapper that uses these prototypes
        class AdaptedModel(nn.Module):
            def __init__(self, embedding_net, prototypes, parent):
                super().__init__()
                self.embedding_net = embedding_net
                self.prototypes = prototypes
                self.parent = parent
            
            def forward(self, x):
                embeddings = self.embedding_net(x)
                return self.parent._predict_from_prototypes(embeddings, self.prototypes)
        
        return AdaptedModel(self.embedding_network, prototypes, self)


class MetaReasoningSystem(nn.Module):
    """
    Complete meta-learning system for first-principles reasoning.
    
    Combines multiple meta-learning approaches and provides a unified
    interface for few-shot reasoning tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        meta_method: str = "maml",
        **meta_kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.meta_method = meta_method
        
        # Create base reasoning network
        self.base_network = ReasoningNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_attention=True
        )
        
        # Create meta-learning system
        if meta_method == "maml":
            self.meta_learner = MAMLReasoningSystem(
                base_network=self.base_network,
                **meta_kwargs
            )
        elif meta_method == "prototypical":
            self.meta_learner = PrototypicalReasoningNetwork(
                embedding_network=self.base_network,
                **meta_kwargs
            )
        else:
            raise ValueError(f"Unknown meta-learning method: {meta_method}")
        
        # Task memory for continual learning
        self.task_memory = {}
        self.memory_size = 1000
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass using base network."""
        return self.base_network(x)
    
    def meta_train(
        self,
        task_batch: TaskBatch,
        n_epochs: int = 1
    ) -> Dict[str, List[float]]:
        """Train the meta-learning system."""
        training_stats = {
            'meta_losses': [],
            'support_losses': [],
            'query_losses': []
        }
        
        for epoch in range(n_epochs):
            # Perform meta-training step
            if hasattr(self.meta_learner, 'meta_train_step'):
                losses = self.meta_learner.meta_train_step(task_batch)
            else:
                losses = self.meta_learner.meta_forward(task_batch)
                # Manual optimization for non-MAML methods
                if 'meta_loss' in losses:
                    loss = losses['meta_loss']
                    # Assuming optimizer is available
                    if hasattr(self, 'optimizer'):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
            
            # Record statistics
            for key, value in losses.items():
                if key in training_stats:
                    training_stats[key].append(value)
        
        return training_stats
    
    def few_shot_adapt(
        self,
        support_examples: Tuple[torch.Tensor, torch.Tensor],
        n_adaptation_steps: int = 10,
        adaptation_lr: float = 0.01
    ) -> nn.Module:
        """
        Rapidly adapt to a new task using few support examples.
        
        Args:
            support_examples: Tuple of (support_x, support_y)
            n_adaptation_steps: Number of adaptation steps
            adaptation_lr: Learning rate for adaptation
            
        Returns:
            Adapted model for the new task
        """
        support_x, support_y = support_examples
        
        # Use meta-learner's adaptation method
        adapted_model = self.meta_learner.adapt_to_task(
            support_x, support_y, n_adaptation_steps
        )
        
        return adapted_model
    
    def continual_learning_update(
        self,
        task_id: str,
        task_data: Tuple[torch.Tensor, torch.Tensor],
        importance_weight: float = 1.0
    ):
        """Update system for continual learning across tasks."""
        x, y = task_data
        
        # Store in task memory
        self.task_memory[task_id] = {
            'data': (x, y),
            'weight': importance_weight,
            'timestamp': torch.tensor(len(self.task_memory))
        }
        
        # Implement memory replay if memory is full
        if len(self.task_memory) > self.memory_size:
            self._memory_consolidation()
    
    def _memory_consolidation(self):
        """Consolidate memory when it becomes full."""
        # Simple strategy: remove oldest tasks with lowest importance
        tasks_by_importance = sorted(
            self.task_memory.items(),
            key=lambda x: x[1]['weight'] * x[1]['timestamp']
        )
        
        # Remove least important task
        task_to_remove = tasks_by_importance[0][0]
        del self.task_memory[task_to_remove]
        
        logger.info(f"Removed task {task_to_remove} from memory during consolidation")
    
    def evaluate_generalization(
        self,
        test_tasks: List[TaskBatch],
        n_shots: List[int] = [1, 5, 10]
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate generalization performance across different shot numbers.
        
        Args:
            test_tasks: List of test tasks
            n_shots: List of shot numbers to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'accuracy': {shot: [] for shot in n_shots},
            'loss': {shot: [] for shot in n_shots}
        }
        
        self.eval()
        with torch.no_grad():
            for task_batch in test_tasks:
                for shot in n_shots:
                    # Take only n_shot examples from support set
                    n_tasks = task_batch.support_x.shape[0]
                    
                    for task_idx in range(n_tasks):
                        support_x = task_batch.support_x[task_idx][:shot]
                        support_y = task_batch.support_y[task_idx][:shot]
                        query_x = task_batch.query_x[task_idx]
                        query_y = task_batch.query_y[task_idx]
                        
                        # Adapt to task
                        adapted_model = self.few_shot_adapt(
                            (support_x, support_y),
                            n_adaptation_steps=5
                        )
                        
                        # Evaluate on query set
                        query_pred = adapted_model(query_x)
                        loss = F.mse_loss(query_pred, query_y)
                        
                        # Compute accuracy (for classification) or R2 score (for regression)
                        if query_y.shape[-1] == 1:  # Regression
                            ss_res = torch.sum((query_y - query_pred) ** 2)
                            ss_tot = torch.sum((query_y - query_y.mean()) ** 2)
                            accuracy = 1 - (ss_res / (ss_tot + 1e-8))
                        else:  # Classification (simplified)
                            accuracy = (torch.argmax(query_pred, dim=1) == torch.argmax(query_y, dim=1)).float().mean()
                        
                        results['accuracy'][shot].append(accuracy.item())
                        results['loss'][shot].append(loss.item())
        
        # Compute means
        for metric in results:
            for shot in n_shots:
                results[metric][shot] = np.mean(results[metric][shot])
        
        return results