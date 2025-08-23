import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import torch.nn as nn
from src.neural_plasticity.neuroplastic_network import AdvancedNeuroplasticNetwork
from src.reinforcement_learning.advanced_rl_agent import AdvancedRLAgent
from src.language_reasoning.language_reasoner import LanguageReasoner
from src.hierarchical_agent.hierarchical_agent import HierarchicalAgent
from src.hybrid_learning.enhanced_hybrid_system import EnhancedHybridLearningSystem
import gymnasium as gym

def run_experiment():
    # Initialize components
    state_dim = 24  # Assuming a complex environment
    action_dim = 4
    goal_dim = 8
    hidden_dim = 256

    neuroplastic_net = AdvancedNeuroplasticNetwork(state_dim + 768, hidden_dim, action_dim)  # 768 is GPT-2's hidden size
    rl_agent = AdvancedRLAgent(hidden_dim, action_dim)
    language_reasoner = LanguageReasoner()
    hierarchical_agent = HierarchicalAgent(state_dim, action_dim, goal_dim, hidden_dim)

    hybrid_system = EnhancedHybridLearningSystem(neuroplastic_net, rl_agent, language_reasoner, hierarchical_agent)

    # Create a complex environment
    env = gym.make('BipedalWalker-v3')  # You can replace this with your custom environment

    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            context = f"Episode: {episode}, Step: {step}, State: {state.tolist()}"
            action, goal, reasoning = hybrid_system.get_action(state, context)
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            hybrid_system.store_transition(state, action, reward, next_state, done, context)

            if step % 10 == 0:  # Update every 10 steps
                hybrid_system.update()

            state = next_state
            step += 1

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        # Periodically train the neuroplastic network and fine-tune the language model
        if episode % 50 == 0:
            states = torch.randn(100, state_dim + 768)  # Mock data
            contexts = torch.randn(100, 768)
            targets = torch.randn(100, action_dim)
            optimizer = optim.Adam(neuroplastic_net.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            hybrid_system.train_neuroplastic_net(states, contexts, targets, optimizer, criterion)

            language_contexts = ["Context " + str(i) for i in range(100)]
            language_responses = torch.randint(0, 50257, (100, 20))  # 50257 is GPT-2's vocab size
            language_optimizer = optim.Adam(language_reasoner.model.parameters(), lr=1e-5)
            language_criterion = nn.CrossEntropyLoss()
            hybrid_system.fine_tune_language_model(language_contexts, language_responses, language_optimizer, language_criterion)

if __name__ == "__main__":
    run_experiment()