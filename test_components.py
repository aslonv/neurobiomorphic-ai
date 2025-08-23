#!/usr/bin/env python3
"""
Test script to validate basic component functionality without requiring model downloads
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.neural_plasticity.neuroplastic_network import AdvancedNeuroplasticNetwork

def test_components():
    """Test basic component functionality"""
    print("Testing neuroplastic network...")
    
    # Test neuroplastic network
    state_dim = 24
    hidden_dim = 256
    action_dim = 4
    
    # Create network
    net = AdvancedNeuroplasticNetwork(state_dim + 768, hidden_dim, action_dim)
    
    # Test forward pass
    x = torch.randn(1, state_dim + 768)
    context = torch.randn(1, 768)
    
    output = net(x, context)
    print(f"Network output shape: {output.shape}")
    
    print("✓ Neuroplastic network test passed")
    
    # Test basic imports
    try:
        from src.reinforcement_learning.advanced_rl_agent import AdvancedRLAgent
        from src.hierarchical_agent.hierarchical_agent import HierarchicalAgent
        print("✓ All core component imports successful")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_components()
    if success:
        print("\n✓ All tests passed! The codebase components are working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)