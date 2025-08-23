import torch

class EnhancedHybridLearningSystem:
    def __init__(self, neuroplastic_net, rl_agent, language_reasoner, hierarchical_agent):
        self.neuroplastic_net = neuroplastic_net
        self.rl_agent = rl_agent
        self.language_reasoner = language_reasoner
        self.hierarchical_agent = hierarchical_agent

    def get_action(self, state, context):
        # Generate reasoning based on context
        reasoning = self.language_reasoner.generate_reasoning(context)
        
        # Extract language features
        language_features = self.language_reasoner.extract_features(reasoning)
        
        # Combine state and language features
        combined_input = torch.cat([torch.FloatTensor(state), language_features.squeeze(0)], dim=-1)
        
        # Pass through neuroplastic network
        neuroplastic_out = self.neuroplastic_net(combined_input.unsqueeze(0), context)
        
        # Get high-level goal from hierarchical agent
        _, goal = self.hierarchical_agent.get_action(neuroplastic_out.squeeze(0).detach().numpy())
        
        # Get low-level action from RL agent
        rl_action, _ = self.rl_agent.get_action(neuroplastic_out.squeeze(0).detach().numpy())
        
        return rl_action, goal, reasoning

    def update(self, batch_size=32):
        self.rl_agent.update(batch_size)
        
        # Update hierarchical agent (assuming we have collected appropriate data)
        self.hierarchical_agent.update_low_level(states, goals, actions, rewards, next_states, dones)
        self.hierarchical_agent.update_high_level(states, goals, rewards, next_states, dones)

    def store_transition(self, state, action, reward, next_state, done, context):
        reasoning = self.language_reasoner.generate_reasoning(context)
        language_features = self.language_reasoner.extract_features(reasoning)
        
        combined_state = torch.cat([torch.FloatTensor(state), language_features.squeeze(0)], dim=-1)
        combined_next_state = torch.cat([torch.FloatTensor(next_state), language_features.squeeze(0)], dim=-1)
        
        neuroplastic_out = self.neuroplastic_net(combined_state.unsqueeze(0), context)
        next_neuroplastic_out = self.neuroplastic_net(combined_next_state.unsqueeze(0), context)
        
        self.rl_agent.store_transition(
            neuroplastic_out.squeeze(0).detach().numpy(),
            action,
            reward,
            next_neuroplastic_out.squeeze(0).detach().numpy(),
            done
        )

    def train_neuroplastic_net(self, states, contexts, targets, optimizer, criterion, num_epochs=10):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.neuroplastic_net(states, contexts)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    def fine_tune_language_model(self, contexts, responses, optimizer, criterion, num_epochs=5):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.language_reasoner.model(contexts)
            loss = criterion(outputs.logits, responses)
            loss.backward()
            optimizer.step()
            print(f"Language Model Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")