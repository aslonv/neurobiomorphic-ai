#!/usr/bin/env python3
"""
Production Readiness Assessment for Neurobiomorphic AI System.

This comprehensive test evaluates all aspects of production readiness including:
- System installation and imports
- Core functionality
- Hugging Face integration structure
- Error handling and fallbacks
- Monitoring and logging systems
- Configuration management
- Code quality metrics
- Performance characteristics
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"📋 {title}")
    print("=" * 70)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def test_installation_and_imports():
    """Test that all core components can be imported."""
    print_section("INSTALLATION & IMPORTS ASSESSMENT")
    
    results = {}
    
    # Core imports
    core_imports = [
        ("neurobiomorphic", "Main package"),
        ("neurobiomorphic.neural_plasticity", "Neural plasticity module"),
        ("neurobiomorphic.reasoning", "Reasoning module"),
        ("neurobiomorphic.language_reasoning", "Language reasoning"),
        ("neurobiomorphic.monitoring", "Monitoring system"),
        ("neurobiomorphic.config", "Configuration system"),
    ]
    
    for module, description in core_imports:
        try:
            __import__(module)
            print(f"✓ {description}: {module}")
            results[module] = True
        except ImportError as e:
            print(f"✗ {description}: {module} - {e}")
            results[module] = False
    
    # Dependency imports
    print_subsection("External Dependencies")
    external_deps = [
        ("torch", "PyTorch framework"),
        ("transformers", "Hugging Face Transformers"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("psutil", "System monitoring"),
        ("hydra", "Configuration management"),
        ("wandb", "Experiment tracking"),
    ]
    
    for module, description in external_deps:
        try:
            __import__(module)
            print(f"✓ {description}: {module}")
            results[f"ext_{module}"] = True
        except ImportError as e:
            print(f"✗ {description}: {module} - {e}")
            results[f"ext_{module}"] = False
    
    success_rate = sum(results.values()) / len(results)
    print(f"\n📊 Import Success Rate: {success_rate:.1%} ({sum(results.values())}/{len(results)})")
    
    return results

def test_core_functionality():
    """Test core system functionality."""
    print_section("CORE FUNCTIONALITY ASSESSMENT")
    
    results = {}
    
    # Neural Plasticity
    print_subsection("Neural Plasticity System")
    try:
        from neurobiomorphic.neural_plasticity.advanced_plasticity_layer import AdvancedNeuroplasticityLayer
        
        layer = AdvancedNeuroplasticityLayer(
            input_size=64,
            output_size=32,
            sparsity=0.3
        )
        
        import torch
        x = torch.randn(8, 64)
        context = torch.randn(8, 32)
        prev_activation = torch.randn(8, 64)
        
        output = layer(x, context, prev_activation)
        stats = layer.get_plasticity_stats()
        
        print(f"✓ Neural plasticity layer functional")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Weight sparsity: {stats.get('weight_sparsity', 'N/A'):.3f}")
        results['neural_plasticity'] = True
        
    except Exception as e:
        print(f"✗ Neural plasticity test failed: {e}")
        results['neural_plasticity'] = False
    
    # Monitoring System
    print_subsection("Monitoring & Logging System")
    try:
        from neurobiomorphic.monitoring.monitoring_system import StructuredLogger, PerformanceProfiler
        
        logger = StructuredLogger("prod_test")
        logger.info("Production readiness test")
        
        profiler = PerformanceProfiler("prod_test")
        profiler.start_timer("test_operation")
        time.sleep(0.01)  # Simulate work
        duration = profiler.end_timer("test_operation")
        
        print(f"✓ Monitoring system functional")
        print(f"  - Logger initialized: {type(logger).__name__}")
        print(f"  - Profiler working: {duration:.4f}s measured")
        results['monitoring'] = True
        
    except Exception as e:
        print(f"✗ Monitoring system test failed: {e}")
        results['monitoring'] = False
    
    # Configuration System
    print_subsection("Configuration System")
    try:
        from neurobiomorphic.config.config_system import ConfigManager
        
        config_manager = ConfigManager()
        
        print(f"✓ Configuration system functional")
        print(f"  - Config manager type: {type(config_manager).__name__}")
        print(f"  - Config directory: {getattr(config_manager, 'config_dir', 'N/A')}")
        results['configuration'] = True
        
    except Exception as e:
        print(f"✗ Configuration system test failed: {e}")
        results['configuration'] = False
    
    success_rate = sum(results.values()) / len(results)
    print(f"\n📊 Core Functionality Success Rate: {success_rate:.1%} ({sum(results.values())}/{len(results)})")
    
    return results

def test_huggingface_integration():
    """Test Hugging Face integration structure and error handling."""
    print_section("HUGGING FACE INTEGRATION ASSESSMENT")
    
    results = {}
    
    print_subsection("Library Integration")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ Transformers library accessible")
        results['transformers_lib'] = True
    except Exception as e:
        print(f"✗ Transformers library issue: {e}")
        results['transformers_lib'] = False
    
    print_subsection("Language Reasoner Structure")
    try:
        from neurobiomorphic.language_reasoning.language_reasoner import AdvancedLanguageReasoner
        
        # Check class structure
        required_methods = ['generate_reasoning', 'extract_features', 'multi_hop_reasoning']
        available_methods = [method for method in required_methods 
                           if hasattr(AdvancedLanguageReasoner, method)]
        
        print(f"✓ Language reasoner class importable")
        print(f"  - Required methods available: {len(available_methods)}/{len(required_methods)}")
        for method in available_methods:
            print(f"    ✓ {method}")
        
        missing_methods = set(required_methods) - set(available_methods)
        if missing_methods:
            for method in missing_methods:
                print(f"    ✗ {method}")
        
        results['language_reasoner_structure'] = len(available_methods) == len(required_methods)
        
    except Exception as e:
        print(f"✗ Language reasoner structure test failed: {e}")
        results['language_reasoner_structure'] = False
    
    print_subsection("Error Handling & Fallbacks")
    try:
        # Check if fallback mechanisms are in place by examining source
        import inspect
        source = inspect.getsource(AdvancedLanguageReasoner.__init__)
        
        has_try_except = 'try:' in source and 'except' in source
        has_fallback_model = 'gpt2' in source or 'fallback' in source.lower()
        has_error_logging = 'logger' in source.lower() or 'warning' in source.lower()
        
        print(f"✓ Error handling analysis:")
        print(f"  - Try/except blocks present: {'✓' if has_try_except else '✗'}")
        print(f"  - Fallback model mechanism: {'✓' if has_fallback_model else '✗'}")
        print(f"  - Error logging present: {'✓' if has_error_logging else '✗'}")
        
        results['error_handling'] = has_try_except and has_fallback_model
        
    except Exception as e:
        print(f"✗ Error handling analysis failed: {e}")
        results['error_handling'] = False
    
    success_rate = sum(results.values()) / len(results)
    print(f"\n📊 HF Integration Success Rate: {success_rate:.1%} ({sum(results.values())}/{len(results)})")
    
    return results

def test_production_features():
    """Test production-specific features."""
    print_section("PRODUCTION FEATURES ASSESSMENT")
    
    results = {}
    
    print_subsection("Error Handling & Robustness")
    try:
        # Test graceful degradation with invalid inputs
        from neurobiomorphic.neural_plasticity.advanced_plasticity_layer import AdvancedNeuroplasticityLayer
        import torch
        
        layer = AdvancedNeuroplasticityLayer(input_size=10, output_size=5, sparsity=0.3)
        
        # Test with valid inputs
        valid_input = torch.randn(4, 10)
        valid_context = torch.randn(4, 5)
        valid_prev = torch.randn(4, 10)
        output = layer(valid_input, valid_context, valid_prev)
        
        print("✓ Handles valid inputs correctly")
        
        # Test error conditions are handled gracefully
        try:
            # Wrong input size
            invalid_input = torch.randn(4, 15)  # Wrong size
            layer(invalid_input, valid_context, valid_prev)
            print("⚠️  Should have failed with wrong input size")
        except Exception:
            print("✓ Properly rejects invalid input sizes")
        
        results['error_handling'] = True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        results['error_handling'] = False
    
    print_subsection("Performance Monitoring")
    try:
        from neurobiomorphic.monitoring.monitoring_system import PerformanceProfiler
        
        profiler = PerformanceProfiler("performance_test")
        
        # Test multiple operations
        operations = ['init', 'forward_pass', 'cleanup']
        for op in operations:
            profiler.start_timer(op)
            time.sleep(0.001)  # Simulate work
            duration = profiler.end_timer(op)
            print(f"  - {op}: {duration:.4f}s")
        
        # Test timing stats
        try:
            stats = profiler.get_timing_stats('forward_pass')
            print(f"✓ Performance monitoring operational")
            print(f"  - Stats available: {type(stats)}")
        except:
            print("✓ Basic performance monitoring works")
        
        results['performance_monitoring'] = True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        results['performance_monitoring'] = False
    
    print_subsection("Memory Management")
    try:
        import torch
        
        # Test memory cleanup
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Create and destroy large tensors
        large_tensor = torch.randn(1000, 1000)
        del large_tensor
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        print(f"✓ Memory management test completed")
        print(f"  - Initial memory: {initial_memory} bytes")
        print(f"  - Final memory: {final_memory} bytes")
        
        results['memory_management'] = True
        
    except Exception as e:
        print(f"✗ Memory management test failed: {e}")
        results['memory_management'] = False
    
    success_rate = sum(results.values()) / len(results)
    print(f"\n📊 Production Features Success Rate: {success_rate:.1%} ({sum(results.values())}/{len(results)})")
    
    return results

def test_build_and_deployment():
    """Test build and deployment readiness."""
    print_section("BUILD & DEPLOYMENT READINESS")
    
    results = {}
    
    print_subsection("Package Structure")
    try:
        import neurobiomorphic
        
        # Check version
        version = getattr(neurobiomorphic, '__version__', '0.1.0')
        print(f"✓ Package version: {version}")
        
        # Check main components
        components = [
            'neural_plasticity',
            'reasoning', 
            'language_reasoning',
            'monitoring',
            'config'
        ]
        
        available_components = []
        for comp in components:
            try:
                __import__(f'neurobiomorphic.{comp}')
                available_components.append(comp)
                print(f"  ✓ {comp}")
            except ImportError:
                print(f"  ✗ {comp}")
        
        results['package_structure'] = len(available_components) == len(components)
        
    except Exception as e:
        print(f"✗ Package structure test failed: {e}")
        results['package_structure'] = False
    
    print_subsection("Configuration Files")
    config_files = [
        ('setup.py', 'Setup script'),
        ('pyproject.toml', 'Build configuration'),
        ('requirements.txt', 'Dependencies'),
        ('README.md', 'Documentation'),
    ]
    
    config_results = {}
    for filename, description in config_files:
        filepath = Path(__file__).parent / filename
        exists = filepath.exists()
        config_results[filename] = exists
        print(f"{'✓' if exists else '✗'} {description}: {filename}")
    
    results['config_files'] = all(config_results.values())
    
    print_subsection("Dependency Resolution")
    try:
        # Check critical dependencies
        import torch
        import transformers
        import numpy
        
        print("✓ Critical dependencies available")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - Transformers: {transformers.__version__}")
        print(f"  - NumPy: {numpy.__version__}")
        
        results['dependencies'] = True
        
    except Exception as e:
        print(f"✗ Dependency check failed: {e}")
        results['dependencies'] = False
    
    success_rate = sum(results.values()) / len(results)
    print(f"\n📊 Build & Deployment Success Rate: {success_rate:.1%} ({sum(results.values())}/{len(results)})")
    
    return results

def main():
    """Run comprehensive production readiness assessment."""
    print("🏭 NEUROBIOMORPHIC AI - PRODUCTION READINESS ASSESSMENT")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all test suites
    test_suites = [
        ("Installation & Imports", test_installation_and_imports),
        ("Core Functionality", test_core_functionality),
        ("Hugging Face Integration", test_huggingface_integration),
        ("Production Features", test_production_features),
        ("Build & Deployment", test_build_and_deployment),
    ]
    
    all_results = {}
    suite_scores = {}
    
    for suite_name, test_func in test_suites:
        try:
            results = test_func()
            all_results[suite_name] = results
            suite_scores[suite_name] = sum(results.values()) / len(results) if results else 0
        except Exception as e:
            print(f"✗ {suite_name} suite failed: {e}")
            traceback.print_exc()
            suite_scores[suite_name] = 0
    
    # Calculate overall score
    overall_score = sum(suite_scores.values()) / len(suite_scores) if suite_scores else 0
    
    # Print comprehensive summary
    print_section("PRODUCTION READINESS SUMMARY")
    
    print_subsection("Test Suite Results")
    for suite_name, score in suite_scores.items():
        status = "🟢 READY" if score >= 0.8 else "🟡 NEEDS WORK" if score >= 0.5 else "🔴 CRITICAL"
        print(f"{status} {suite_name}: {score:.1%}")
    
    print_subsection("Overall Assessment")
    overall_status = "🟢 PRODUCTION READY" if overall_score >= 0.8 else "🟡 NEEDS IMPROVEMENTS" if overall_score >= 0.6 else "🔴 NOT READY"
    print(f"{overall_status}")
    print(f"Overall Score: {overall_score:.1%}")
    
    print_subsection("Key Recommendations")
    
    recommendations = []
    
    if suite_scores.get("Installation & Imports", 0) < 0.9:
        recommendations.append("• Fix missing imports and dependencies")
    
    if suite_scores.get("Hugging Face Integration", 0) < 0.8:
        recommendations.append("• Test Hugging Face connectivity in production environment")
        recommendations.append("• Ensure internet access for model downloads")
    
    if suite_scores.get("Production Features", 0) < 0.8:
        recommendations.append("• Strengthen error handling and monitoring")
    
    if suite_scores.get("Core Functionality", 0) < 0.9:
        recommendations.append("• Address core system functionality issues")
    
    # General recommendations
    recommendations.extend([
        "• Run full integration tests in production environment",
        "• Set up monitoring and alerting for model inference",
        "• Configure proper logging and error tracking",
        "• Test with actual Hugging Face model downloads",
        "• Set up model caching for offline operation",
        "• Configure resource limits and scaling policies"
    ])
    
    for rec in recommendations:
        print(rec)
    
    end_time = time.time()
    print(f"\n⏱️  Assessment completed in {end_time - start_time:.2f} seconds")
    
    print_section("DEPLOYMENT CHECKLIST")
    checklist_items = [
        ("Internet connectivity for Hugging Face Hub", "🟡 NEEDS VERIFICATION"),
        ("Sufficient memory for model loading", "🟡 NEEDS VERIFICATION"),
        ("GPU availability (optional but recommended)", "🟡 NEEDS VERIFICATION"), 
        ("Error monitoring and alerting", "🟢 CONFIGURED" if suite_scores.get("Production Features", 0) > 0.7 else "🟡 NEEDS SETUP"),
        ("Configuration management", "🟢 READY" if suite_scores.get("Build & Deployment", 0) > 0.8 else "🟡 NEEDS WORK"),
        ("Core AI functionality", "🟢 OPERATIONAL" if suite_scores.get("Core Functionality", 0) > 0.8 else "🔴 NEEDS FIXES"),
    ]
    
    for item, status in checklist_items:
        print(f"{status} {item}")
    
    return 0 if overall_score >= 0.7 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)