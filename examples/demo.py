#!/usr/bin/env python3
"""
Example: Neurobiomorphic AI System Demo

This script demonstrates the key capabilities of the neurobiomorphic AI system:
- Neural plasticity mechanisms
- Uncertainty quantification
- Causal reasoning
- Language reasoning

Run with: python examples/demo.py
"""

import torch
import numpy as np
from neurobiomorphic.neural_plasticity import AdvancedNeuroplasticNetwork
from neurobiomorphic.reasoning.uncertainty_quantification import BayesianNeuralNetwork
from neurobiomorphic.reasoning.causal_engine import CausalReasoningEngine
from neurobiomorphic.language_reasoning.language_reasoner import AdvancedLanguageReasoner


def demo_neuroplastic_network():
    """Demonstrate neural plasticity capabilities."""
    print("=== Neural Plasticity Demo ===")
    
    # Create a neuroplastic network
    network = AdvancedNeuroplasticNetwork(
        input_size=10,
        hidden_size=64, 
        output_size=5,
        num_layers=3,
        adaptive_architecture=True
    )
    
    # Generate sample data
    batch_size = 32
    x = torch.randn(batch_size, 10)
    context = torch.randn(batch_size, 64)
    
    # Forward pass with plasticity
    output = network(x, context)
    print(f"Neuroplastic network output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in network.parameters()):,}")
    print()


def demo_uncertainty_quantification():
    """Demonstrate uncertainty quantification.""" 
    print("=== Uncertainty Quantification Demo ===")
    
    # Create Bayesian neural network
    model = BayesianNeuralNetwork(
        input_dim=5,
        hidden_dims=[32, 16],
        output_dim=1
    )
    
    # Generate sample data
    x = torch.randn(16, 5)
    
    # Get uncertainty estimates
    estimate = model.forward_with_uncertainty(x, n_samples=50)
    
    print(f"Prediction shape: {estimate.prediction.shape}")
    print(f"Mean epistemic uncertainty: {estimate.epistemic_uncertainty.mean().item():.4f}")
    print(f"Mean aleatoric uncertainty: {estimate.aleatoric_uncertainty.mean().item():.4f}")
    print(f"Mean total uncertainty: {estimate.total_uncertainty.mean().item():.4f}")
    print()


def demo_causal_reasoning():
    """Demonstrate causal reasoning capabilities."""
    print("=== Causal Reasoning Demo ===")
    
    # Create causal reasoning engine
    variables = ["X", "Y", "Z"]
    mechanism_configs = {
        "X": {"input_dim": 3, "hidden_dims": [8], "output_dim": 1},
        "Y": {"input_dim": 3, "hidden_dims": [8], "output_dim": 1}, 
        "Z": {"input_dim": 3, "hidden_dims": [8], "output_dim": 1}
    }
    
    try:
        causal_engine = CausalReasoningEngine(
            variable_names=variables,
            mechanism_configs=mechanism_configs
        )
        
        print(f"Causal engine created with {len(variables)} variables")
        print(f"Available mechanisms: {list(causal_engine.mechanisms.keys())}")
        print(f"Adjacency matrix shape: {causal_engine.adj_matrix.shape}")
        
    except Exception as e:
        print(f"Causal reasoning demo simplified (complex graph): {str(e)[:100]}...")
    
    print()


def demo_language_reasoning():
    """Demonstrate language reasoning capabilities."""
    print("=== Language Reasoning Demo ===")
    
    try:
        # Create language reasoner
        reasoner = AdvancedLanguageReasoner(
            model_name="microsoft/DialoGPT-medium",
            use_chain_of_thought=True,
            use_working_memory=True
        )
        
        print("Language reasoner created successfully")
        print(f"Model name: {reasoner.model_name}")
        print(f"Hidden dimensions: {reasoner.hidden_dim}")
        print("Features: Chain-of-thought, Working memory, Causal intervention")
        
    except Exception as e:
        print(f"Language reasoner demo skipped (missing dependencies): {e}")
    
    print()


def main():
    """Run all demonstrations."""
    print("🧠 Neurobiomorphic AI System Demo")
    print("="*50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstrations
    demo_neuroplastic_network()
    demo_uncertainty_quantification() 
    demo_causal_reasoning()
    demo_language_reasoning()
    
    print("✅ Demo completed successfully!")
    print("\nThis framework provides:")
    print("- Advanced neural plasticity with biological mechanisms")
    print("- Robust uncertainty quantification for reliable AI") 
    print("- First-principles causal reasoning capabilities")
    print("- Sophisticated language understanding and reasoning")
    print("\nPerfect for researchers working on:")
    print("- Continual learning systems")
    print("- Causal AI and reasoning")
    print("- Uncertainty-aware AI")
    print("- Biologically-inspired neural networks")


if __name__ == "__main__":
    main()