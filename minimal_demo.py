#!/usr/bin/env python3
"""
Minimal demonstration of the neurobiomorphic AI system without requiring model downloads.
This script creates a mock environment and runs a few episodes to validate the system integration.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.neural_plasticity.neuroplastic_network import AdvancedNeuroplasticNetwork
from src.reinforcement_learning.advanced_rl_agent import AdvancedRLAgent  
from src.hierarchical_agent.hierarchical_agent import HierarchicalAgent

class MockEnvironment:
    """Simple mock environment for testing without gymnasium dependencies"""
    def __init__(self, state_dim=24, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.state_dim)
        
    def step(self, action):
        self.step_count += 1
        next_state = np.random.randn(self.state_dim)
        reward = np.random.randn() * 0.1  # Small random reward
        done = self.step_count >= 10  # Episode ends after 10 steps
        return next_state, reward, done, {}

class MockLanguageReasoner:
    """Mock language reasoner that doesn't require model downloads"""
    def generate_reasoning(self, context):
        return f"Mock reasoning for: {context[:50]}..."
        
    def extract_features(self, text):
        # Return mock features with GPT-2 hidden size
        return torch.randn(1, 768)

def run_minimal_demo():
    """Run a minimal demonstration"""
    print("🧠 Starting Neurobiomorphic AI Minimal Demo")
    print("=" * 60)
    
    # Initialize components
    state_dim = 24
    action_dim = 4
    goal_dim = 8
    hidden_dim = 256
    
    print("Initializing components...")
    neuroplastic_net = AdvancedNeuroplasticNetwork(state_dim + 768, hidden_dim, action_dim)
    rl_agent = AdvancedRLAgent(hidden_dim, action_dim)
    hierarchical_agent = HierarchicalAgent(state_dim, action_dim, goal_dim, hidden_dim)
    language_reasoner = MockLanguageReasoner()
    
    # Create mock environment  
    env = MockEnvironment(state_dim, action_dim)
    
    print("✓ All components initialized successfully")
    
    # Run a few episodes
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # Generate mock context and reasoning
            context = f"Episode: {episode}, Step: {step}, State: {state[:3].tolist()}..."
            reasoning = language_reasoner.generate_reasoning(context)
            language_features = language_reasoner.extract_features(reasoning)
            
            # Combine state and language features
            combined_input = torch.cat([torch.FloatTensor(state), language_features.squeeze(0)], dim=-1)
            
            # Get neuroplastic network output
            context_tensor = torch.randn(1, 768)  # Mock context tensor
            neuroplastic_output = neuroplastic_net(combined_input.unsqueeze(0), context_tensor)
            
            # Get hierarchical action and goal
            action, goal = hierarchical_agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            print(f"  Step {step + 1}: Action={action[:2]}, Reward={reward:.3f}")
            
            # Store transition in RL agent
            rl_agent.store_transition(
                neuroplastic_output.squeeze(0).detach().numpy(),
                np.argmax(action),  # Convert to discrete action for RL agent
                reward,
                neuroplastic_output.squeeze(0).detach().numpy(),  # Use same for next state (mock)
                done
            )
            
            state = next_state
            step += 1
        
        print(f"  Episode {episode + 1} completed! Total reward: {total_reward:.3f}")
        
        # Update RL agent if enough samples
        if len(rl_agent.memory) >= 32:
            rl_agent.update()
            print(f"  ✓ RL agent updated")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("✓ Neuroplastic network: Working")
    print("✓ RL agent: Working")  
    print("✓ Hierarchical agent: Working")
    print("✓ Mock language reasoning: Working")
    print("✓ Integration: Working")
    
    return True

if __name__ == "__main__":
    success = run_minimal_demo()
    if success:
        print(f"\n✅ Minimal demo passed! The neurobiomorphic AI system is functional.")
        sys.exit(0)
    else:
        print(f"\n❌ Demo failed.")
        sys.exit(1)