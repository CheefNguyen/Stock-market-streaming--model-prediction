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
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device = 'cuda'
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.9  # Higher prioritize long-term rewards/ Low short-term
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 # Higher can do more risky move to explore, Lower lead to sooner convergance
        self.epsilon_decay = 0.985
        self.learning_rate = 0.005

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)
        self.loss_fn = nn.MSELoss()
        self.prev_performance = None


    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()
        self.model.train()  # Set the model back to training mode
        return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).view(1, -1)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device).view(1, -1)
            target = reward

            if not done:
                with torch.no_grad():
                    next_q_values = self.target_model(next_state_tensor)
                    target += self.gamma * torch.max(next_q_values).item()

            state_tensor = state_tensor.view(-1)  # Flatten state tensor
            target_f = self.model(state_tensor).detach().clone()
            target_f[action] = target

            states.append(state_tensor)
            targets_f.append(target_f)

        states = torch.stack(states)
        targets_f = torch.stack(targets_f)

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.loss_fn(outputs, targets_f)
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()
        return loss.item()

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