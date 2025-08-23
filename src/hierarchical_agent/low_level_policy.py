import torch
import torch.nn as nn
import torch.nn.functional as F

class LowLevelPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):  # Changed from hidden_dim to goal_dim
        super().__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, goal_dim)  # Use goal_dim instead of hidden_dim
        self.fc2 = nn.Linear(goal_dim, goal_dim)
        self.fc3 = nn.Linear(goal_dim, action_dim)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)