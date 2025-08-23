#!/usr/bin/env python3
"""
Comprehensive integration test for the Neurobiomorphic AI system.

This test verifies that all major components work together correctly
and demonstrates the system's capabilities.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_neural_plasticity():
    """Test the advanced neural plasticity system."""
    print("\n=== Testing Neural Plasticity System ===")
    
    try:
        from neurobiomorphic.neural_plasticity.advanced_plasticity_layer import AdvancedNeuroplasticityLayer
        from neurobiomorphic.neural_plasticity.neuroplastic_network import AdvancedNeuroplasticNetwork
        
        # Create plasticity layer
        layer = AdvancedNeuroplasticityLayer(
            input_size=128,
            output_size=64,
            sparsity=0.7,
            enable_metaplasticity=True,
            enable_structural_plasticity=True
        )
        
        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, 128)
        context = torch.randn(batch_size, 64)
        prev_activation = torch.randn(batch_size, 128)
        
        output = layer(x, context, prev_activation)
        assert output.shape == (batch_size, 64), f"Expected shape {(batch_size, 64)}, got {output.shape}"
        
        # Test plasticity statistics
        stats = layer.get_plasticity_stats()
        assert 'mean_synaptic_tag' in stats
        assert 'weight_sparsity' in stats
        
        print(f"✓ Neural plasticity layer test passed")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Weight sparsity: {stats['weight_sparsity']:.3f}")
        
        # Test full network
        network = AdvancedNeuroplasticNetwork(
            input_size=128,
            hidden_size=64,
            output_size=32,
            num_layers=3,
            adaptive_architecture=True
        )
        
        output = network(x)
        assert output.shape == (batch_size, 32)
        
        network_stats = network.get_network_stats()
        assert 'network_sparsity' in network_stats
        
        print(f"✓ Neuroplastic network test passed")
        print(f"  - Network sparsity: {network_stats['network_sparsity']:.3f}")
        print(f"  - Effective capacity: {network.get_effective_capacity():.3f}")
        
    except Exception as e:
        print(f"✗ Neural plasticity test failed: {e}")
        return False
    
    return True


def test_causal_reasoning():
    """Test the causal reasoning system."""
    print("\n=== Testing Causal Reasoning System ===")
    
    try:
        from neurobiomorphic.reasoning.causal_engine import CausalReasoningEngine, CausalGraph
        
        # Define simple causal system
        variables = ["X", "Y", "Z"]
        mechanisms = {
            "Y": {
                "input_dim": 1,
                "hidden_dims": [32],
                "output_dim": 1,
                "use_bayesian": True
            },
            "Z": {
                "input_dim": 2,
                "hidden_dims": [32],
                "output_dim": 1,
                "use_bayesian": True
            }
        }
        
        engine = CausalReasoningEngine(
            variable_names=variables,
            mechanism_configs=mechanisms
        )
        
        # Test with synthetic data
        batch_size = 100
        data = torch.randn(batch_size, 3)
        
        # Test causal discovery
        loss, metrics = engine.causal_discovery_loss(data)
        assert torch.is_tensor(loss)
        assert 'reconstruction' in metrics
        
        print(f"✓ Causal discovery test passed")
        print(f"  - Discovery loss: {loss.item():.4f}")
        print(f"  - Reconstruction loss: {metrics['reconstruction']:.4f}")
        
        # Test causal graph extraction
        graph = engine.get_causal_graph(threshold=0.3)
        assert isinstance(graph, CausalGraph)
        assert len(graph.nodes) == 3
        
        print(f"✓ Causal graph test passed")
        print(f"  - Number of edges: {len(graph.edges)}")
        
        # Test interventions
        interventions = {"X": torch.tensor([1.0])}
        generated, _ = engine.forward(data[:10], interventions)
        assert generated.shape == (10, 3)
        
        print(f"✓ Causal intervention test passed")
        
    except Exception as e:
        print(f"✗ Causal reasoning test failed: {e}")
        return False
    
    return True


def test_meta_learning():
    """Test the meta-learning system."""
    print("\n=== Testing Meta-Learning System ===")
    
    try:
        from neurobiomorphic.reasoning.meta_learning import MetaReasoningSystem, TaskBatch
        
        # Create meta-learning system
        meta_system = MetaReasoningSystem(
            input_dim=64,
            output_dim=10,
            meta_method="maml"
        )
        
        # Create synthetic task batch
        n_tasks = 4
        n_support = 5
        n_query = 10
        
        task_batch = TaskBatch(
            support_x=torch.randn(n_tasks, n_support, 64),
            support_y=torch.randn(n_tasks, n_support, 10),
            query_x=torch.randn(n_tasks, n_query, 64),
            query_y=torch.randn(n_tasks, n_query, 10)
        )
        
        # Test meta-training
        stats = meta_system.meta_train(task_batch, n_epochs=3)
        assert 'meta_losses' in stats
        
        print(f"✓ Meta-learning training test passed")
        print(f"  - Meta-training completed with {len(stats['meta_losses'])} epochs")
        
        # Test few-shot adaptation
        support_x = torch.randn(n_support, 64)
        support_y = torch.randn(n_support, 10)
        
        adapted_model = meta_system.few_shot_adapt(
            (support_x, support_y),
            n_adaptation_steps=5
        )
        
        # Test adapted model
        test_x = torch.randn(5, 64)
        predictions = adapted_model(test_x)
        assert predictions.shape == (5, 10)
        
        print(f"✓ Few-shot adaptation test passed")
        print(f"  - Adapted model prediction shape: {predictions.shape}")
        
    except Exception as e:
        print(f"✗ Meta-learning test failed: {e}")
        return False
    
    return True


def test_uncertainty_quantification():
    """Test the uncertainty quantification system."""
    print("\n=== Testing Uncertainty Quantification ===")
    
    try:
        from neurobiomorphic.reasoning.uncertainty_quantification import (
            BayesianNeuralNetwork,
            MCDropoutNetwork,
            UncertaintyAggregator
        )
        
        # Create Bayesian Neural Network
        bnn = BayesianNeuralNetwork(
            input_dim=64,
            hidden_dims=[32, 16],
            output_dim=1
        )
        
        # Test uncertainty estimation
        test_data = torch.randn(10, 64)
        uncertainty_estimate = bnn.forward_with_uncertainty(test_data, n_samples=50)
        
        assert hasattr(uncertainty_estimate, 'prediction')
        assert hasattr(uncertainty_estimate, 'epistemic_uncertainty')
        assert hasattr(uncertainty_estimate, 'aleatoric_uncertainty')
        
        print(f"✓ Bayesian neural network test passed")
        print(f"  - Prediction shape: {uncertainty_estimate.prediction.shape}")
        print(f"  - Mean epistemic uncertainty: {uncertainty_estimate.epistemic_uncertainty.mean().item():.4f}")
        
        # Test MC Dropout
        mc_dropout = MCDropoutNetwork(
            input_dim=64,
            hidden_dims=[32, 16],
            output_dim=1,
            dropout_rate=0.2
        )
        
        mc_uncertainty = mc_dropout.forward_with_uncertainty(test_data, n_samples=50)
        assert mc_uncertainty.prediction.shape == uncertainty_estimate.prediction.shape
        
        print(f"✓ MC Dropout test passed")
        
        # Test uncertainty aggregation
        aggregator = UncertaintyAggregator([bnn, mc_dropout])
        aggregated = aggregator.aggregate_uncertainties(test_data)
        
        assert aggregated.prediction.shape == uncertainty_estimate.prediction.shape
        
        print(f"✓ Uncertainty aggregation test passed")
        
    except Exception as e:
        print(f"✗ Uncertainty quantification test failed: {e}")
        return False
    
    return True


def test_monitoring_system():
    """Test the monitoring and logging system."""
    print("\n=== Testing Monitoring System ===")
    
    try:
        from neurobiomorphic.monitoring.monitoring_system import (
            StructuredLogger,
            SystemMonitor,
            PerformanceProfiler,
            ExperimentTracker
        )
        
        # Test structured logging
        logger = StructuredLogger("test_component")
        logger.set_context(experiment="test", version="1.0")
        
        logger.info("Test log message", metric=0.95)
        logger.warning("Test warning", threshold=0.8)
        
        print(f"✓ Structured logging test passed")
        
        # Test system monitoring (without starting the monitoring thread)
        monitor = SystemMonitor(enable_gpu_monitoring=False)
        current_metrics = monitor._collect_metrics()
        
        assert hasattr(current_metrics, 'cpu_percent')
        assert hasattr(current_metrics, 'memory_percent')
        assert current_metrics.cpu_percent >= 0
        
        print(f"✓ System monitoring test passed")
        print(f"  - CPU usage: {current_metrics.cpu_percent:.1f}%")
        print(f"  - Memory usage: {current_metrics.memory_percent:.1f}%")
        
        # Test performance profiling
        profiler = PerformanceProfiler("test_profiler")
        
        with profiler.time_operation("test_operation"):
            # Simulate some work
            torch.randn(1000, 1000).sum()
        
        stats = profiler.get_timing_stats("test_operation")
        assert 'mean' in stats
        assert stats['count'] == 1
        
        profiler.record_custom_metric("test_metric", 0.85)
        report = profiler.generate_report()
        
        assert 'timing_stats' in report
        assert 'custom_metrics' in report
        
        print(f"✓ Performance profiling test passed")
        print(f"  - Test operation time: {stats['mean']:.4f}s")
        
    except Exception as e:
        print(f"✗ Monitoring system test failed: {e}")
        return False
    
    return True


def test_configuration_system():
    """Test the configuration management system."""
    print("\n=== Testing Configuration System ===")
    
    try:
        from neurobiomorphic.config.config_system import ConfigManager, FullConfig
        
        # Create config manager
        config_manager = ConfigManager("/tmp/test_configs")
        
        # Create and save a test configuration
        config = FullConfig()
        config.base.name = "test_experiment"
        config.model.input_dim = 128
        config.model.output_dim = 10
        
        config_manager.save_config(config, "test_config")
        
        # Load configuration
        loaded_config = config_manager.load_config("test_config")
        
        print(f"✓ Configuration management test passed")
        print(f"  - Config name: {loaded_config.base.name}")
        print(f"  - Model input dim: {loaded_config.model.input_dim}")
        
        # Test validation
        errors = config_manager.validate_config(config)
        # Some errors are expected due to missing required fields
        
        print(f"  - Validation errors found: {len(errors)}")
        
    except Exception as e:
        print(f"✗ Configuration system test failed: {e}")
        return False
    
    return True


def test_integration():
    """Test integration of multiple components."""
    print("\n=== Testing System Integration ===")
    
    try:
        from neurobiomorphic.neural_plasticity.neuroplastic_network import AdvancedNeuroplasticNetwork
        from neurobiomorphic.reasoning.uncertainty_quantification import BayesianNeuralNetwork
        from neurobiomorphic.monitoring.monitoring_system import PerformanceProfiler
        
        # Create integrated system
        profiler = PerformanceProfiler("integration_test")
        
        with profiler.time_operation("system_initialization"):
            # Create neuroplastic network
            network = AdvancedNeuroplasticNetwork(
                input_size=64,
                hidden_size=32,
                output_size=16,
                num_layers=2
            )
            
            # Create uncertainty quantification
            uncertainty_net = BayesianNeuralNetwork(
                input_dim=64,
                hidden_dims=[32],
                output_dim=16
            )
        
        # Test combined forward pass
        test_input = torch.randn(8, 64)
        
        with profiler.time_operation("forward_pass"):
            # Neuroplastic network forward
            plastic_output = network(test_input)
            
            # Uncertainty estimation
            uncertainty_est = uncertainty_net.forward_with_uncertainty(test_input)
        
        # Verify outputs
        assert plastic_output.shape == (8, 16)
        assert uncertainty_est.prediction.shape == (8, 16)
        
        # Record performance metrics
        profiler.record_custom_metric("output_variance", plastic_output.var().item())
        profiler.record_custom_metric("mean_uncertainty", uncertainty_est.total_uncertainty.mean().item())
        
        # Generate performance report
        report = profiler.generate_report()
        
        print(f"✓ System integration test passed")
        print(f"  - Plastic network output shape: {plastic_output.shape}")
        print(f"  - Uncertainty prediction shape: {uncertainty_est.prediction.shape}")
        print(f"  - Mean total uncertainty: {uncertainty_est.total_uncertainty.mean().item():.4f}")
        
        # Print performance summary
        init_time = report['timing_stats']['system_initialization']['mean']
        forward_time = report['timing_stats']['forward_pass']['mean']
        
        print(f"  - System initialization time: {init_time:.4f}s")
        print(f"  - Forward pass time: {forward_time:.4f}s")
        
    except Exception as e:
        print(f"✗ System integration test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests and report results."""
    print("🧠 Neurobiomorphic AI System - Comprehensive Integration Test")
    print("=" * 60)
    
    # List of all tests
    tests = [
        test_neural_plasticity,
        test_causal_reasoning,
        test_meta_learning,
        test_uncertainty_quantification,
        test_monitoring_system,
        test_configuration_system,
        test_integration
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} encountered unexpected error: {e}")
            failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"📊 Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The Neurobiomorphic AI system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)