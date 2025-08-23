"""
Advanced Uncertainty Quantification System

Implements multiple uncertainty estimation techniques for robust reasoning:
- Bayesian Neural Networks with variational inference
- Deep ensembles for predictive uncertainty
- Monte Carlo Dropout for epistemic uncertainty
- Evidential Deep Learning for aleatoric/epistemic decomposition
- Conformal prediction for distribution-free uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates with different types."""
    prediction: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    total_uncertainty: torch.Tensor
    confidence_interval: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    prediction_sets: Optional[List[torch.Tensor]] = None


class UncertaintyQuantificationBase(nn.Module, ABC):
    """Base class for uncertainty quantification methods."""
    
    @abstractmethod
    def forward_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> UncertaintyEstimate:
        """Forward pass that returns prediction with uncertainty estimates."""
        pass
    
    @abstractmethod
    def calibrate(self, cal_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Calibrate uncertainty estimates using validation data."""
        pass


class BayesianLinearLayer(nn.Module):
    """Bayesian linear layer with variational inference."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        posterior_rho_init: float = -3.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * posterior_rho_init)
        
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * posterior_rho_init)
        
        self.register_buffer('prior_weight_mu', torch.zeros(out_features, in_features))
        self.register_buffer('prior_weight_std', torch.ones(out_features, in_features) * prior_std)
        self.register_buffer('prior_bias_mu', torch.zeros(out_features))
        self.register_buffer('prior_bias_std', torch.ones(out_features) * prior_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reparameterization trick."""
        weight_std = torch.log(1 + torch.exp(self.weight_rho))
        bias_std = torch.log(1 + torch.exp(self.bias_rho))
        
        if self.training:
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        weight_std = torch.log(1 + torch.exp(self.weight_rho))
        weight_kl = self._gaussian_kl(
            self.weight_mu, weight_std, 
            self.prior_weight_mu, self.prior_weight_std
        )
        
        bias_std = torch.log(1 + torch.exp(self.bias_rho))
        bias_kl = self._gaussian_kl(
            self.bias_mu, bias_std,
            self.prior_bias_mu, self.prior_bias_std
        )
        
        return weight_kl + bias_kl
    
    def _gaussian_kl(self, mu_q, std_q, mu_p, std_p):
        """KL divergence between two Gaussians."""
        return torch.sum(
            torch.log(std_p / std_q) + 
            (std_q**2 + (mu_q - mu_p)**2) / (2 * std_p**2) - 0.5
        )
    
    def sample_weights(self, n_samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample multiple weight realizations."""
        weight_std = torch.log(1 + torch.exp(self.weight_rho))
        bias_std = torch.log(1 + torch.exp(self.bias_rho))
        
        samples = []
        for _ in range(n_samples):
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
            samples.append((weight, bias))
        
        return samples


class BayesianNeuralNetwork(UncertaintyQuantificationBase):
    """Bayesian Neural Network with variational inference."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_std: float = 1.0,
        kl_weight: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kl_weight = kl_weight
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinearLayer(prev_dim, hidden_dim, prior_std))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(BayesianLinearLayer(prev_dim, output_dim, prior_std))
        
        self.layers = nn.ModuleList([layer for layer in layers if isinstance(layer, BayesianLinearLayer)])
        self.activations = nn.ModuleList([layer for layer in layers if not isinstance(layer, BayesianLinearLayer)])
        
        self.log_var_head = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning mean prediction."""
        layer_idx = 0
        activation_idx = 0
        
        for layer in self.layers[:-1]:
            x = layer(x)
            if activation_idx < len(self.activations):
                x = self.activations[activation_idx](x)
                activation_idx += 1
            layer_idx += 1
        
        x = self.layers[-1](x)
        return x
    
    def forward_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> UncertaintyEstimate:
        """Forward pass with uncertainty quantification."""
        batch_size = x.shape[0]
        device = x.device
        
        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)
        
        with torch.no_grad():
            features = self._extract_features(x)
            log_var = self.log_var_head(features)
            aleatoric_uncertainty = torch.exp(log_var)
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        std_total = torch.sqrt(total_uncertainty)
        lower_ci = mean_pred - 1.96 * std_total
        upper_ci = mean_pred + 1.96 * std_total
        
        return UncertaintyEstimate(
            prediction=mean_pred,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_interval=(lower_ci, upper_ci)
        )
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from second-to-last layer."""
        layer_idx = 0
        activation_idx = 0
        
        for layer in self.layers[:-1]:
            x = layer(x)
            if activation_idx < len(self.activations):
                x = self.activations[activation_idx](x)
                activation_idx += 1
        
        return x
    
    def kl_loss(self) -> torch.Tensor:
        """Compute total KL divergence loss."""
        kl_div = 0.0
        for layer in self.layers:
            kl_div += layer.kl_divergence()
        return self.kl_weight * kl_div
    
    def elbo_loss(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Evidence Lower Bound (ELBO) loss."""
        # Likelihood term
        pred = self.forward(x)
        
        # Aleatoric uncertainty
        features = self._extract_features(x)
        log_var = self.log_var_head(features)
        
        # Heteroscedastic loss
        precision = torch.exp(-log_var)
        likelihood_loss = 0.5 * (precision * (y - pred)**2 + log_var).mean()
        
        # KL divergence term
        kl_loss = self.kl_loss()
        
        # Total ELBO loss
        total_loss = likelihood_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'likelihood_loss': likelihood_loss,
            'kl_loss': kl_loss
        }
    
    def calibrate(self, cal_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Calibrate uncertainty using temperature scaling."""
        cal_x, cal_y = cal_data
        
        # Temperature parameter
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                uncertainty_est = self.forward_with_uncertainty(cal_x, n_samples=50)
                logits = uncertainty_est.prediction
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            loss = F.cross_entropy(scaled_logits, cal_y)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Store calibrated temperature
        self.temperature = temperature.item()
        logger.info(f"Calibrated temperature: {self.temperature}")


class DeepEnsemble(UncertaintyQuantificationBase):
    """Deep Ensemble for uncertainty quantification."""
    
    def __init__(
        self,
        base_network_factory: Callable[[], nn.Module],
        n_models: int = 5,
        diversity_regularization: float = 0.1
    ):
        super().__init__()
        self.n_models = n_models
        self.diversity_regularization = diversity_regularization
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            base_network_factory() for _ in range(n_models)
        ])
        
        # Individual optimizers for each model
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=0.001)
            for model in self.models
        ]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using ensemble average."""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=0).mean(dim=0)
    
    def forward_with_uncertainty(self, x: torch.Tensor, n_samples: int = None) -> UncertaintyEstimate:
        """Forward pass with uncertainty from ensemble disagreement."""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [n_models, batch_size, output_dim]
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)
        
        # For deep ensembles, epistemic uncertainty is the primary source
        aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)
        total_uncertainty = epistemic_uncertainty
        
        # Confidence intervals
        std_total = torch.sqrt(total_uncertainty)
        lower_ci = mean_pred - 1.96 * std_total
        upper_ci = mean_pred + 1.96 * std_total
        
        return UncertaintyEstimate(
            prediction=mean_pred,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_interval=(lower_ci, upper_ci)
        )
    
    def train_ensemble(
        self,
        train_loader,
        n_epochs: int = 100,
        use_diversity_loss: bool = True
    ):
        """Train the ensemble with diversity regularization."""
        for epoch in range(n_epochs):
            total_losses = []
            
            for batch_x, batch_y in train_loader:
                batch_losses = []
                predictions = []
                
                # Forward pass for all models
                for i, model in enumerate(self.models):
                    model.train()
                    pred = model(batch_x)
                    predictions.append(pred)
                    
                    # Standard loss
                    loss = F.mse_loss(pred, batch_y)
                    batch_losses.append(loss)
                
                # Diversity regularization
                if use_diversity_loss and len(predictions) > 1:
                    diversity_loss = self._compute_diversity_loss(predictions)
                    for i in range(len(batch_losses)):
                        batch_losses[i] = batch_losses[i] - self.diversity_regularization * diversity_loss
                
                # Backward pass for each model
                for i, (model, loss, optimizer) in enumerate(zip(self.models, batch_losses, self.optimizers)):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                total_losses.extend([loss.item() for loss in batch_losses])
            
            if epoch % 10 == 0:
                avg_loss = np.mean(total_losses)
                logger.info(f"Ensemble training epoch {epoch}, average loss: {avg_loss:.4f}")
    
    def _compute_diversity_loss(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Compute diversity regularization term."""
        n_models = len(predictions)
        diversity_loss = 0.0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Cosine similarity between predictions
                sim = F.cosine_similarity(predictions[i].flatten(), predictions[j].flatten(), dim=0)
                diversity_loss += sim ** 2
        
        return diversity_loss / (n_models * (n_models - 1) / 2)
    
    def calibrate(self, cal_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Calibrate ensemble predictions."""
        # For deep ensembles, calibration is typically done via temperature scaling
        # on the ensemble average, similar to BNN
        pass


class MCDropoutNetwork(UncertaintyQuantificationBase):
    """Monte Carlo Dropout for uncertainty quantification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Build network with dropout layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.network(x)
    
    def forward_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> UncertaintyEstimate:
        """Forward pass with MC Dropout uncertainty."""
        # Enable dropout during inference
        self.train()
        
        predictions = []
        for _ in range(n_samples):
            pred = self.network(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [n_samples, batch_size, output_dim]
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)
        
        # MC Dropout primarily captures epistemic uncertainty
        aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)
        total_uncertainty = epistemic_uncertainty
        
        # Confidence intervals
        std_total = torch.sqrt(total_uncertainty)
        lower_ci = mean_pred - 1.96 * std_total
        upper_ci = mean_pred + 1.96 * std_total
        
        return UncertaintyEstimate(
            prediction=mean_pred,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_interval=(lower_ci, upper_ci)
        )
    
    def calibrate(self, cal_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Calibrate MC Dropout uncertainty."""
        # Similar to other methods, can use temperature scaling
        pass


class EvidentialDeepLearning(UncertaintyQuantificationBase):
    """
    Evidential Deep Learning for uncertainty quantification.
    
    Models predictions as evidence for a higher-order distribution,
    enabling natural separation of aleatoric and epistemic uncertainty.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        evidence_activation: str = "relu"
    ):
        super().__init__()
        self.output_dim = output_dim
        self.evidence_activation = evidence_activation
        
        # Base network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads for evidential parameters
        self.evidence_head = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning mean prediction."""
        features = self.feature_extractor(x)
        evidence = self._get_evidence(features)
        alpha = evidence + 1
        
        # For Normal-Inverse-Gamma, mean is sum of alphas
        return alpha / alpha.sum(dim=-1, keepdim=True)
    
    def forward_with_uncertainty(self, x: torch.Tensor, n_samples: int = None) -> UncertaintyEstimate:
        """Forward pass with evidential uncertainty decomposition."""
        features = self.feature_extractor(x)
        evidence = self._get_evidence(features)
        alpha = evidence + 1
        
        # Compute Dirichlet parameters
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Mean prediction (Dirichlet expectation)
        mean_pred = alpha / S
        
        # Aleatoric uncertainty (expected data variance)
        aleatoric_uncertainty = (alpha * (S - alpha)) / (S * S * (S + 1))
        
        # Epistemic uncertainty (distributional uncertainty)
        epistemic_uncertainty = alpha / (S * S * (S + 1))
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        # Confidence intervals based on concentration parameters
        concentration = alpha.sum(dim=-1)
        precision = concentration / (concentration + self.output_dim)
        
        std_total = torch.sqrt(total_uncertainty)
        lower_ci = mean_pred - 1.96 * std_total
        upper_ci = mean_pred + 1.96 * std_total
        
        return UncertaintyEstimate(
            prediction=mean_pred,
            epistemic_uncertainty=epistemic_uncertainty.sum(dim=-1, keepdim=True),
            aleatoric_uncertainty=aleatoric_uncertainty.sum(dim=-1, keepdim=True),
            total_uncertainty=total_uncertainty.sum(dim=-1, keepdim=True),
            confidence_interval=(lower_ci, upper_ci)
        )
    
    def _get_evidence(self, features: torch.Tensor) -> torch.Tensor:
        """Convert features to evidence using specified activation."""
        evidence = self.evidence_head(features)
        
        if self.evidence_activation == "relu":
            return F.relu(evidence)
        elif self.evidence_activation == "exp":
            return torch.exp(torch.clamp(evidence, -10, 10))  # Clamp for stability
        elif self.evidence_activation == "softplus":
            return F.softplus(evidence)
        else:
            raise ValueError(f"Unknown evidence activation: {self.evidence_activation}")
    
    def evidential_loss(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 0) -> Dict[str, torch.Tensor]:
        """Compute evidential loss with KL annealing."""
        features = self.feature_extractor(x)
        evidence = self._get_evidence(features)
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Likelihood loss
        likelihood_loss = torch.sum(
            (y - alpha / S) ** 2 + 
            alpha * (S - alpha) / (S * S * (S + 1)), 
            dim=-1
        ).mean()
        
        # KL divergence loss (regularization)
        kl_div = self._kl_divergence(alpha, y)
        
        # Annealing factor (increase regularization over time)
        annealing_factor = min(1.0, epoch / 100.0)
        
        total_loss = likelihood_loss + annealing_factor * kl_div
        
        return {
            'total_loss': total_loss,
            'likelihood_loss': likelihood_loss,
            'kl_loss': kl_div
        }
    
    def _kl_divergence(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for evidential loss."""
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Target alpha (assuming uniform prior)
        alpha_hat = y * (S - self.output_dim) + 1
        
        # KL divergence between Dirichlet distributions
        kl_div = torch.lgamma(alpha_hat.sum(dim=-1)) - torch.lgamma(alpha.sum(dim=-1))
        kl_div += torch.sum(
            torch.lgamma(alpha) - torch.lgamma(alpha_hat) + 
            (alpha_hat - alpha) * (torch.digamma(alpha_hat) - torch.digamma(alpha_hat.sum(dim=-1, keepdim=True))),
            dim=-1
        )
        
        return kl_div.mean()
    
    def calibrate(self, cal_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Calibrate evidential predictions."""
        # Evidential models are naturally calibrated, but can still benefit from post-hoc calibration
        pass


class ConformalPrediction:
    """
    Conformal Prediction for distribution-free uncertainty quantification.
    
    Provides finite-sample guarantees for prediction intervals/sets
    without distributional assumptions.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        alpha: float = 0.1  # Miscoverage level (1-alpha confidence)
    ):
        self.base_model = base_model
        self.alpha = alpha
        self.quantiles = None
        self.is_calibrated = False
    
    def calibrate(self, cal_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Calibrate conformal predictor using calibration data."""
        cal_x, cal_y = cal_data
        
        self.base_model.eval()
        with torch.no_grad():
            predictions = self.base_model(cal_x)
        
        # Compute conformity scores (absolute residuals for regression)
        scores = torch.abs(predictions - cal_y)
        
        # Compute quantile for prediction intervals
        n_cal = len(scores)
        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        self.quantiles = torch.quantile(scores, quantile_level, dim=0)
        
        self.is_calibrated = True
        logger.info(f"Conformal predictor calibrated with quantile: {quantile_level}")
    
    def predict_with_sets(self, x: torch.Tensor) -> UncertaintyEstimate:
        """Generate conformal prediction sets."""
        if not self.is_calibrated:
            raise ValueError("Conformal predictor must be calibrated first")
        
        self.base_model.eval()
        with torch.no_grad():
            predictions = self.base_model(x)
        
        # Create prediction intervals
        lower_bound = predictions - self.quantiles
        upper_bound = predictions + self.quantiles
        
        # For conformal prediction, uncertainty is interval width
        interval_width = upper_bound - lower_bound
        total_uncertainty = interval_width / 4  # Rough approximation to std
        
        return UncertaintyEstimate(
            prediction=predictions,
            epistemic_uncertainty=torch.zeros_like(total_uncertainty),  # Not decomposed
            aleatoric_uncertainty=torch.zeros_like(total_uncertainty),   # Not decomposed
            total_uncertainty=total_uncertainty,
            confidence_interval=(lower_bound, upper_bound)
        )


class UncertaintyAggregator:
    """
    Aggregates uncertainty estimates from multiple methods
    for robust uncertainty quantification.
    """
    
    def __init__(
        self,
        uncertainty_methods: List[UncertaintyQuantificationBase],
        aggregation_strategy: str = "ensemble"
    ):
        self.uncertainty_methods = uncertainty_methods
        self.aggregation_strategy = aggregation_strategy
        self.method_weights = torch.ones(len(uncertainty_methods))
    
    def aggregate_uncertainties(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100
    ) -> UncertaintyEstimate:
        """Aggregate uncertainty estimates from multiple methods."""
        estimates = []
        
        for method in self.uncertainty_methods:
            estimate = method.forward_with_uncertainty(x, n_samples)
            estimates.append(estimate)
        
        if self.aggregation_strategy == "ensemble":
            return self._ensemble_aggregate(estimates)
        elif self.aggregation_strategy == "weighted":
            return self._weighted_aggregate(estimates)
        elif self.aggregation_strategy == "stacking":
            return self._stacking_aggregate(estimates)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
    
    def _ensemble_aggregate(self, estimates: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """Simple ensemble averaging of estimates."""
        n_methods = len(estimates)
        
        mean_pred = torch.stack([est.prediction for est in estimates]).mean(dim=0)
        
        epistemic = torch.stack([est.epistemic_uncertainty for est in estimates]).mean(dim=0)
        aleatoric = torch.stack([est.aleatoric_uncertainty for est in estimates]).mean(dim=0)
        total = torch.stack([est.total_uncertainty for est in estimates]).mean(dim=0)
        
        lower_bounds = torch.stack([est.confidence_interval[0] for est in estimates if est.confidence_interval])
        upper_bounds = torch.stack([est.confidence_interval[1] for est in estimates if est.confidence_interval])
        
        if len(lower_bounds) > 0:
            lower_ci = lower_bounds.min(dim=0)[0]
            upper_ci = upper_bounds.max(dim=0)[0] 
            confidence_interval = (lower_ci, upper_ci)
        else:
            confidence_interval = None
        
        return UncertaintyEstimate(
            prediction=mean_pred,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence_interval=confidence_interval
        )
    
    def _weighted_aggregate(self, estimates: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """Weighted aggregation based on method weights."""
        weights = F.softmax(self.method_weights, dim=0)
        
        mean_pred = sum(w * est.prediction for w, est in zip(weights, estimates))
        
        epistemic = sum(w * est.epistemic_uncertainty for w, est in zip(weights, estimates))
        aleatoric = sum(w * est.aleatoric_uncertainty for w, est in zip(weights, estimates))
        total = sum(w * est.total_uncertainty for w, est in zip(weights, estimates))
        
        return UncertaintyEstimate(
            prediction=mean_pred,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total
        )
    
    def _stacking_aggregate(self, estimates: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """Stacking-based aggregation (requires meta-learning)."""
        # This would require training a meta-model to combine estimates
        # For now, fall back to ensemble
        return self._ensemble_aggregate(estimates)
    
    def calibrate_weights(
        self,
        val_data: Tuple[torch.Tensor, torch.Tensor],
        metric: str = "mse"
    ):
        """Calibrate method weights based on validation performance."""
        val_x, val_y = val_data
        method_scores = []
        
        for method in self.uncertainty_methods:
            estimate = method.forward_with_uncertainty(val_x)
            
            if metric == "mse":
                score = F.mse_loss(estimate.prediction, val_y)
            elif metric == "nll":
                std = torch.sqrt(estimate.total_uncertainty)
                nll = 0.5 * torch.log(2 * np.pi * std**2) + 0.5 * (estimate.prediction - val_y)**2 / std**2
                score = nll.mean()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            method_scores.append(score)
        
        scores_tensor = torch.stack(method_scores)
        weights = 1.0 / (scores_tensor + 1e-8)
        self.method_weights = weights / weights.sum()
        
        logger.info(f"Calibrated method weights: {self.method_weights}")


def uncertainty_based_active_learning(
    model: UncertaintyQuantificationBase,
    pool_x: torch.Tensor,
    n_select: int = 10,
    strategy: str = "uncertainty"
) -> torch.Tensor:
    """
    Active learning based on uncertainty estimates.
    
    Args:
        model: Uncertainty quantification model
        pool_x: Unlabeled data pool
        n_select: Number of samples to select
        strategy: Selection strategy ("uncertainty", "entropy", "bald")
    
    Returns:
        Indices of selected samples
    """
    model.eval()
    uncertainties = []
    
    with torch.no_grad():
        for i in range(0, len(pool_x), 32):
            batch = pool_x[i:i+32]
            estimate = model.forward_with_uncertainty(batch)
            
            if strategy == "uncertainty":
                batch_uncertainty = estimate.total_uncertainty.sum(dim=1)
            elif strategy == "entropy":
                probs = F.softmax(estimate.prediction, dim=1)
                batch_uncertainty = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            elif strategy == "bald":
                batch_uncertainty = estimate.epistemic_uncertainty.sum(dim=1)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            uncertainties.append(batch_uncertainty)
    
    all_uncertainties = torch.cat(uncertainties, dim=0)
    
    _, selected_indices = torch.topk(all_uncertainties, n_select)
    
    return selected_indices