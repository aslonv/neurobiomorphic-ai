import torch
import torch.nn as nn
import torch.optim as optim

class HierarchicalAgent:
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.low_level_policy = LowLevelPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.high_level_policy = HighLevelPolicy(state_dim, goal_dim, hidden_dim).to(self.device)
        self.low_level_optimizer = optim.Adam(self.low_level_policy.parameters(), lr=lr)
        self.high_level_optimizer = optim.Adam(self.high_level_policy.parameters(), lr=lr)
        self.goal_dim = goal_dim
        self.action_dim = action_dim

    def get_action(self, state, goal=None):
        state = torch.FloatTensor(state).to(self.device)
        if goal is None:
            goal = self.high_level_policy(state)
        action = self.low_level_policy(state, goal)
        return action.cpu().detach().numpy(), goal.cpu().detach().numpy()

    def update_low_level(self, states, goals, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        predicted_actions = self.low_level_policy(states, goals)
        loss = nn.MSELoss()(predicted_actions, actions)

        self.low_level_optimizer.zero_grad()
        loss.backward()
        self.low_level_optimizer.step()

        return loss.item()

    def update_high_level(self, states, goals, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        predicted_goals = self.high_level_policy(states)
        next_goals = self.high_level_policy(next_states)

        # Assuming a simple TD learning approach for the high-level policy
        target_goals = rewards + (1 - dones) * 0.99 * next_goals
        loss = nn.MSELoss()(predicted_goals, target_goals.detach())

        self.high_level_optimizer.zero_grad()
        loss.backward()
        self.high_level_optimizer.step()

        return loss.item()