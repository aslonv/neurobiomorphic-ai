"""
Production-Grade Configuration System

Implements a comprehensive configuration management system using Hydra and OmegaConf
for reproducible experiments and production deployments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from omegaconf import MISSING, OmegaConf
import os
from pathlib import Path

# Base configuration for all components
@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    name: str = MISSING
    version: str = "1.0.0"
    description: str = ""
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps
    precision: str = "float32"  # float16, float32, float64
    deterministic: bool = True
    

@dataclass 
class ModelConfig:
    """Configuration for neural network models."""
    architecture: str = MISSING
    input_dim: int = MISSING
    output_dim: int = MISSING
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dropout_rate: float = 0.1
    normalization: str = "layer_norm"  # batch_norm, layer_norm, none
    initialization: str = "xavier"  # xavier, kaiming, normal
    
    
@dataclass
class ReasoningConfig:
    """Configuration for reasoning systems."""
    causal_reasoning: Dict[str, Any] = field(default_factory=dict)
    symbolic_neural_hybrid: Dict[str, Any] = field(default_factory=dict)
    meta_learning: Dict[str, Any] = field(default_factory=dict)
    uncertainty_quantification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalReasoningConfig:
    """Configuration for causal reasoning engine."""
    discovery_method: str = "notears"
    max_parents: int = 5
    mechanism_type: str = "neural"  # neural, linear, nonlinear
    use_bayesian: bool = True
    sparsity_penalty: float = 0.01
    dag_penalty: float = 10.0
    temperature: float = 1.0
    n_samples: int = 100


@dataclass 
class SymbolicNeuralConfig:
    """Configuration for symbolic-neural hybrid system."""
    vocab_size: int = 1000
    embedding_dim: int = 256
    hidden_dim: int = 512
    max_expression_length: int = 64
    n_transformer_layers: int = 3
    n_attention_heads: int = 8
    supported_operations: List[str] = field(
        default_factory=lambda: ["add", "multiply", "compose", "conditional"]
    )
    rule_memory_size: int = 100


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning systems."""
    method: str = "maml"  # maml, prototypical, meta_sgd
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    n_inner_steps: int = 5
    n_support: int = 5
    n_query: int = 15
    n_tasks_per_batch: int = 8
    first_order: bool = False
    

@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""
    methods: List[str] = field(
        default_factory=lambda: ["bayesian", "ensemble", "mc_dropout"]
    )
    n_samples: int = 100
    n_ensemble_members: int = 5
    bayesian_prior_std: float = 1.0
    mc_dropout_rate: float = 0.1
    evidential_activation: str = "relu"
    conformal_alpha: float = 0.1
    aggregation_strategy: str = "ensemble"


@dataclass
class NeuralPlasticityConfig:
    """Configuration for neural plasticity systems."""
    plasticity_type: str = "advanced"  # basic, advanced, continual
    consolidation_rate: float = 0.001
    tag_decay: float = 0.99
    protein_synthesis_threshold: float = 0.5
    homeostatic_target: float = 0.1
    astrocyte_modulation: bool = True
    dendritic_computation: bool = True
    stdp_learning_rate: float = 0.01
    metaplasticity: bool = True
    synaptic_scaling: bool = True


@dataclass
class ReinforcementLearningConfig:
    """Configuration for RL agents."""
    algorithm: str = "ppo"  # ppo, sac, td3, dqn
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    entropy_coefficient: float = 0.01


@dataclass
class TrainingConfig:
    """Configuration for training procedures."""
    optimizer: str = "adam"  # adam, adamw, sgd, rmsprop
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # cosine, exponential, plateau, none
    warmup_steps: int = 1000
    max_epochs: int = 1000
    early_stopping_patience: int = 50
    gradient_clip_value: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = False
    

@dataclass
class DataConfig:
    """Configuration for data handling."""
    dataset_name: str = MISSING
    data_dir: str = "data/"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    validation_split: float = 0.1
    test_split: float = 0.1
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    augmentation: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class LoggingConfig:
    """Configuration for logging and monitoring."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_dir: str = "logs/"
    tensorboard_dir: str = "runs/"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 10
    log_model_params: bool = False
    log_gradients: bool = False
    

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    name: str = MISSING
    description: str = ""
    tags: List[str] = field(default_factory=list)
    output_dir: str = "outputs/"
    checkpoint_dir: str = "checkpoints/"
    resume_from_checkpoint: Optional[str] = None
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"  # min, max
    

@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    method: str = "optuna"  # optuna, ray_tune, grid_search
    n_trials: int = 100
    timeout: Optional[int] = None
    pruning: bool = True
    objective_metric: str = "val_loss"
    objective_direction: str = "minimize"  # minimize, maximize
    search_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionConfig:
    """Configuration for production deployment."""
    model_registry: str = "local"  # local, mlflow, wandb
    serving_framework: str = "torchserve"  # torchserve, triton, onnx
    batch_inference: bool = False
    max_batch_size: int = 32
    timeout_seconds: int = 30
    health_check_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"
    model_versioning: bool = True
    

@dataclass
class FullConfig:
    """Complete configuration combining all components."""
    base: BaseConfig = field(default_factory=BaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    causal_reasoning: CausalReasoningConfig = field(default_factory=CausalReasoningConfig)
    symbolic_neural: SymbolicNeuralConfig = field(default_factory=SymbolicNeuralConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    neural_plasticity: NeuralPlasticityConfig = field(default_factory=NeuralPlasticityConfig)
    reinforcement_learning: ReinforcementLearningConfig = field(default_factory=ReinforcementLearningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    hyperparameter: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)


class ConfigManager:
    """Manages configuration loading, validation, and updates."""
    
    def __init__(self, config_dir: str = "configs/"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
    def load_config(self, config_name: str) -> FullConfig:
        """Load configuration from file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if config_path.exists():
            conf = OmegaConf.load(config_path)
            # Convert to structured config
            return OmegaConf.to_container(conf, resolve=True)
        else:
            # Return default config
            return FullConfig()
    
    def save_config(self, config: FullConfig, config_name: str) -> None:
        """Save configuration to file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        # Convert to OmegaConf and save
        conf = OmegaConf.structured(config)
        OmegaConf.save(conf, config_path)
        
    def merge_configs(self, base_config: FullConfig, override_config: Dict[str, Any]) -> FullConfig:
        """Merge override config into base config."""
        base_conf = OmegaConf.structured(base_config)
        override_conf = OmegaConf.create(override_config)
        merged = OmegaConf.merge(base_conf, override_conf)
        return OmegaConf.to_container(merged, resolve=True)
    
    def validate_config(self, config: FullConfig) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate required fields
        if config.base.name == MISSING:
            errors.append("base.name is required")
        if config.model.input_dim == MISSING:
            errors.append("model.input_dim is required") 
        if config.model.output_dim == MISSING:
            errors.append("model.output_dim is required")
        if config.data.dataset_name == MISSING:
            errors.append("data.dataset_name is required")
        if config.experiment.name == MISSING:
            errors.append("experiment.name is required")
            
        # Validate ranges
        if config.training.learning_rate <= 0:
            errors.append("training.learning_rate must be positive")
        if config.data.batch_size <= 0:
            errors.append("data.batch_size must be positive")
        if config.training.max_epochs <= 0:
            errors.append("training.max_epochs must be positive")
            
        # Validate choices
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if config.base.device not in valid_devices:
            errors.append(f"base.device must be one of {valid_devices}")
            
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop"]
        if config.training.optimizer not in valid_optimizers:
            errors.append(f"training.optimizer must be one of {valid_optimizers}")
            
        return errors
    
    def get_experiment_config(
        self,
        experiment_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> FullConfig:
        """Get configuration for a specific experiment."""
        # Load base config
        config = self.load_config("default")
        
        # Apply experiment-specific overrides
        if overrides:
            config = self.merge_configs(config, overrides)
            
        # Set experiment name
        config.experiment.name = experiment_name
        
        # Validate configuration
        errors = self.validate_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
            
        return config
    
    def create_default_configs(self) -> None:
        """Create default configuration files."""
        default_config = FullConfig()
        self.save_config(default_config, "default")
        
        # Create example experiment configs
        reasoning_config = FullConfig()
        reasoning_config.base.name = "reasoning_experiment"
        reasoning_config.base.description = "First-principles reasoning experiment"
        reasoning_config.model.architecture = "reasoning_network"
        reasoning_config.experiment.tags = ["reasoning", "causal", "symbolic"]
        self.save_config(reasoning_config, "reasoning_experiment")
        
        # Meta-learning experiment
        meta_config = FullConfig()
        meta_config.base.name = "meta_learning_experiment"  
        meta_config.base.description = "Few-shot meta-learning experiment"
        meta_config.model.architecture = "meta_network"
        meta_config.meta_learning.method = "maml"
        meta_config.experiment.tags = ["meta_learning", "few_shot"]
        self.save_config(meta_config, "meta_learning_experiment")
        
        # Uncertainty quantification experiment
        uncertainty_config = FullConfig()
        uncertainty_config.base.name = "uncertainty_experiment"
        uncertainty_config.base.description = "Uncertainty quantification experiment"
        uncertainty_config.uncertainty.methods = ["bayesian", "ensemble", "mc_dropout"]
        uncertainty_config.experiment.tags = ["uncertainty", "bayesian", "ensemble"]
        self.save_config(uncertainty_config, "uncertainty_experiment")


def setup_environment_from_config(config: FullConfig) -> None:
    """Setup environment variables and paths from configuration."""
    import torch
    import numpy as np
    import random
    
    # Set random seeds for reproducibility
    if config.base.deterministic:
        torch.manual_seed(config.base.seed)
        np.random.seed(config.base.seed)
        random.seed(config.base.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.base.seed)
            torch.cuda.manual_seed_all(config.base.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    os.makedirs(config.data.data_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.tensorboard_dir, exist_ok=True)
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)
    
    # Set device
    if config.base.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.base.device
    
    # Set environment variables
    os.environ['NEUROBIOMORPHIC_DEVICE'] = device
    os.environ['NEUROBIOMORPHIC_PRECISION'] = config.base.precision


def get_config_schema() -> Dict[str, Any]:
    """Get the configuration schema for documentation/validation."""
    return OmegaConf.structured(FullConfig).to_yaml()


# Initialize default config manager
default_config_manager = ConfigManager()

# Create default configs if they don't exist
if not (default_config_manager.config_dir / "default.yaml").exists():
    default_config_manager.create_default_configs()