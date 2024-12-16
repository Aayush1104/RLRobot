import numpy as np
import torch
import torch.nn as nn

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0003, discount_factor=0.99, epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.hidden_size = 128
        
        # Initialize neural network for Q-learning
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def get_action(self, state, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            return 2 * np.random.rand(self.action_dim) - 1
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.q_network(state_tensor).numpy()[0]
            return np.clip(action, -1, 1)

    def update_q_table(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        
        # Calculate current Q value
        current_q = self.q_network(state_tensor)
        
        # Calculate target Q value
        with torch.no_grad():
            next_q = self.q_network(next_state_tensor)
            target_q = reward_tensor + self.gamma * next_q.max(1)[0]
        
        # Update network
        loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
