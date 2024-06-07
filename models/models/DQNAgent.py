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
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99  # Higher prioritize long-term rewards/ Low short-term
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1 # Higher can do more risky move to explore, Lower lead to sooner convergance
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)
        self.loss_fn = nn.MSELoss()
        self.prev_performance = None


    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def forward(self, x):
        return self.model(x)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, shares_held):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)

        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.model(state_tensor).detach().cpu().numpy()
        action = np.argmax(q_values)
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0.0
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).view(1, -1)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device).view(1, -1)
            target = torch.tensor(reward, dtype=torch.float32).to(self.device)

            if not done:
                next_q_values = self.target_model(next_state_tensor).detach()
                target += self.gamma * torch.max(next_q_values)

            target_f = self.model(state_tensor).detach().clone()
            target_f[0, action] = target

            self.model.zero_grad()
            loss = self.loss_fn(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        self.scheduler.step()

        return total_loss

    def performance_update_epsilon_decay(self, performance_metrics):
        current_profit, rewards = performance_metrics
        if self.prev_performance is not None:
            prev_profit, prev_rewards = self.prev_performance

            profit_change_ratio = (current_profit - prev_profit) / max(1, abs(prev_profit))
            rewards_change_ratio = (rewards - prev_rewards) / max(1, abs(prev_rewards))

            profit_change_threshold = 0.2
            rewards_change_threshold = 0.2

            if (profit_change_ratio > profit_change_threshold and
                rewards_change_ratio > rewards_change_threshold):
                self.epsilon_decay *= 0.99
            elif (profit_change_ratio < -0.6 and
                  rewards_change_ratio < -0.6):
                self.epsilon_decay *= 1.01

        self.prev_performance = performance_metrics
        self.epsilon_decay = min(max(self.epsilon_decay, self.epsilon_min), 1.0)

    def calculate_correct_action_rate(self, total_correct_actions, total_actions):
        if total_actions == 0:
            return 0
        return (total_correct_actions / total_actions) * 100


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_agent(self, model_weights_filename):
        torch.save(self.model.state_dict(), model_weights_filename)

    def load_agent(self, model_weights_filename):
        self.model.load_state_dict(torch.load(model_weights_filename))
        self.target_model.load_state_dict(torch.load(model_weights_filename))

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay