import numpy as np
import random

from collections import deque

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam

import pickle

import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device = 'cuda'
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Higher prioritize long-term rewards/ Low short-term
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 # Higher can do more risky move to explore, Lower lead to sooner convergance
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001

        self.model = self._build_model().to(self.device)  # Move model to GPU if available
        self.target_model = self._build_model().to(self.device)  # Move target model to GPU if available
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model with model's weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.prev_performance = None


    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def forward(self, x):
        return self.model(x)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)  # Move state to GPU if available
        q_values = self.model(state_tensor.unsqueeze(0)).detach().cpu().numpy()  # Detach and move output to CPU
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0.0
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor.unsqueeze(0)))
            target_f = self.model(state_tensor.unsqueeze(0))
            target_f[0] = target
            loss = self.loss_fn(target_f, self.model(state_tensor.unsqueeze(0)))
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss

    def performance_update_epsilon_decay(self, performance_metrics):
        current_reward, current_profit, win_rate = performance_metrics

        if self.prev_performance is not None:
            prev_reward, prev_profit, prev_win_rate = self.prev_performance
        
            reward_change_ratio = (current_reward - prev_reward) / max(1, abs(prev_reward))
            profit_change_ratio = (current_profit - prev_profit) / max(1, abs(prev_profit))
            win_rate_change_ratio = (win_rate - prev_win_rate) / max(1, abs(prev_win_rate))

            # Define thresholds for significant changes
            reward_change_threshold = 0.2  # Adjust as needed
            profit_change_threshold = 0.2  # Adjust as needed
            win_rate_change_threshold = 0.2  # Adjust as needed

            # Adjust epsilon decay based on performance changes
            if (reward_change_ratio > reward_change_threshold and
                profit_change_ratio > profit_change_threshold and
                win_rate_change_ratio > win_rate_change_threshold):
                # Significant improvement in reward, profit, and win rate
                self.epsilon_decay *= 0.99  # Decrease epsilon decay rate
            elif (reward_change_ratio < -reward_change_threshold and
                profit_change_ratio < -profit_change_threshold and
                win_rate_change_ratio < -win_rate_change_threshold):
                # Significant decline in reward, profit, and win rate
                self.epsilon_decay *= 1.01  # Increase epsilon decay rate

        self.prev_performance = performance_metrics
        self.epsilon_decay = max(self.epsilon_decay, self.epsilon_min)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model_weights(self, filename):
        torch.save(self.target_model.state_dict(), filename)

    def save_agent_state(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.memory, self.gamma, self.epsilon), f)

    def save_agent(self, model_weights_filename, agent_state_filename):
        self.save_model_weights(model_weights_filename)
        self.save_agent_state(agent_state_filename)

    def load_agent(self, model_weights_filename, agent_state_filename):
        self.load_model_weights(model_weights_filename)
        self.load_agent_state(agent_state_filename)
    
    def load_model_weights(self, filename):
        self.target_model.load_state_dict(torch.load(filename))

    def load_agent_state(self, filename):
        with open(filename, 'rb') as f:
            self.memory, self.gamma, self.epsilon = pickle.load(f)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_sell_rate(self, rewards):
        # successful_episodes tang 1 khi mang rewards co gia tri != 0
        # => cang thuc hien lenh ban nhieu thi win rate tang
        # => khong dung dk cua win rate
        successful_episodes = sum(1 for reward in rewards if reward > 0)
        total_episodes = len(rewards)
        win_rate = (successful_episodes / total_episodes) * 100
        return win_rate
