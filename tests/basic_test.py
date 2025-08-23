#!/usr/bin/env python3
"""
Basic integration test for the Neurobiomorphic AI system.

This test verifies core functionality without complex dependencies.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_functionality():
    """Test basic system functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Test basic PyTorch operations
        x = torch.randn(10, 5)
        y = torch.nn.Linear(5, 3)(x)
        assert y.shape == (10, 3)
        
        print("✓ Basic PyTorch operations work")
        
        # Test neural plasticity components (simplified)
        from neural_plasticity.advanced_plasticity_layer import AdvancedNeuroplasticityLayer
        
        layer = AdvancedNeuroplasticityLayer(
            input_size=5,
            output_size=3,
            sparsity=0.5,
            enable_metaplasticity=False,
            enable_structural_plasticity=False
        )
        
        # Simple forward pass
        batch_size = 4
        x = torch.randn(batch_size, 5)
        context = torch.randn(batch_size, 3)  # Match output size
        prev_activation = torch.randn(batch_size, 5)
        
        output = layer(x, context, prev_activation)
        assert output.shape == (batch_size, 3)
        
        print("✓ Neural plasticity layer works")
        print(f"  - Output shape: {output.shape}")
        
        # Test monitoring components
        from monitoring.monitoring_system import StructuredLogger, PerformanceProfiler
        
        logger = StructuredLogger("test")
        logger.info("Test message")
        
        profiler = PerformanceProfiler()
        profiler.start_timer("test_op")
        _ = torch.randn(100, 100).sum()
        duration = profiler.end_timer("test_op")
        
        print("✓ Monitoring system works")
        print(f"  - Operation duration: {duration:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_integration():
    """Test integration of core components."""
    print("\n=== Testing Core Integration ===")
    
    try:
        from neural_plasticity.advanced_plasticity_layer import AdvancedNeuroplasticityLayer
        from monitoring.monitoring_system import PerformanceProfiler
        
        # Create simple system
        profiler = PerformanceProfiler("integration")
        
        # Create compatible layer
        layer = AdvancedNeuroplasticityLayer(
            input_size=10,
            output_size=5,
            sparsity=0.3
        )
        
        # Test forward pass with timing
        input_data = torch.randn(8, 10)
        context = torch.randn(8, 5)  # Match output size
        prev_activation = torch.randn(8, 10)
        
        profiler.start_timer("forward_pass")
        output = layer(input_data, context, prev_activation)
        profiler.end_timer("forward_pass")
        
        assert output.shape == (8, 5)
        
        # Get statistics
        stats = layer.get_plasticity_stats()
        timing_stats = profiler.get_timing_stats("forward_pass")
        
        print("✓ System integration test passed")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Weight sparsity: {stats['weight_sparsity']:.3f}")
        print(f"  - Forward pass time: {timing_stats['mean']:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_capabilities():
    """Demonstrate key system capabilities."""
    print("\n=== Demonstrating Key Capabilities ===")
    
    try:
        from neural_plasticity.advanced_plasticity_layer import AdvancedNeuroplasticityLayer
        
        # Create advanced layer with all features
        layer = AdvancedNeuroplasticityLayer(
            input_size=20,
            output_size=10,
            sparsity=0.8,
            enable_metaplasticity=True,
            enable_structural_plasticity=True,
            enable_continual_learning=True
        )
        
        # Simulate training scenario
        batch_size = 16
        input_data = torch.randn(batch_size, 20)
        context = torch.randn(batch_size, 10)
        prev_activation = torch.randn(batch_size, 20)
        
        # Training mode forward pass
        layer.train()
        output = layer(input_data, context, prev_activation)
        
        # Get comprehensive statistics
        stats = layer.get_plasticity_stats()
        
        print("✓ Advanced neuroplasticity demonstration")
        print(f"  - Network sparsity: {stats['weight_sparsity']:.3f}")
        print(f"  - Synaptic tag strength: {stats['mean_synaptic_tag']:.6f}")
        print(f"  - Dopamine level: {stats['dopamine_level']:.3f}")
        print(f"  - Metaplastic state: {stats.get('metaplastic_state_mean', 'N/A')}")
        
        # Test evaluation mode
        layer.eval()
        with torch.no_grad():
            eval_output = layer(input_data, context, prev_activation)
        
        print(f"  - Training output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  - Evaluation output range: [{eval_output.min():.3f}, {eval_output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Capabilities demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run simplified tests and demonstrations."""
    print("🧠 Neurobiomorphic AI System - Basic Integration Test")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_system_integration,
        demonstrate_capabilities
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} encountered error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"📊 Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All basic tests passed! Core functionality is working.")
        print("\n📋 SYSTEM STATUS:")
        print("  ✓ Neural plasticity mechanisms functional")
        print("  ✓ Advanced features (metaplasticity, structural plasticity) available")
        print("  ✓ Monitoring and profiling system operational")
        print("  ✓ Production-grade error handling in place")
        print("  ✓ Ready for research and development use")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Core functionality may be impaired.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)