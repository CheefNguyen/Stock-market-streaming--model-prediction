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
    def __init__(self, state_size, action_size, num_tickers):
        self.device = 'cuda'
        self.state_size = state_size
        self.action_size = action_size
        self.num_tickers = num_tickers
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Higher prioritize long-term rewards/ Low short-term
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001 # Higher can do more risky move to explore, Lower lead to sooner convergance
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
            nn.Linear(64, self.action_size * self.num_tickers)
        )
        return model
    
    def forward(self, x):
        return self.model(x)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.action_size) for _ in range(self.num_tickers)]
        
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.model(state_tensor).detach().cpu().numpy()
        q_values = q_values.reshape(self.num_tickers, self.action_size)  # Ensure correct shape for slicing
        
        actions = [np.argmax(q_values[i]) for i in range(self.num_tickers)]
        return actions

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

            # Ensure action is a list of integers
            if isinstance(action, list):
                for i in range(self.num_tickers):
                    target_f[0, i * self.action_size + action[i]] = target
            else:
                target_f[0, action] = target  # Fallback if action is not a list

            self.model.zero_grad()
            loss = self.loss_fn(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss

    def performance_update_epsilon_decay(self, performance_metrics):
        current_reward, correct_action_rate = performance_metrics
        if self.prev_performance is not None:
            """
            When total_reward increase and profit increase 
            -> decrease epsilon daycay for faster convergence
            """
            prev_reward, prev_correct_action_rate = self.prev_performance

            reward_change_ratio = (current_reward - prev_reward) / max(1, abs(prev_reward))
            correct_action_rate_change_ratio = (correct_action_rate - prev_correct_action_rate) / max(1, abs(prev_correct_action_rate))

            reward_change_threshold = 0.2
            correct_action_rate_change_threshold = 0.2

            if (reward_change_ratio > reward_change_threshold and
                correct_action_rate_change_ratio > correct_action_rate_change_threshold):
                # If performance of episode improves, decrease epsilon decay rate
                self.epsilon_decay *= 0.99

            elif (reward_change_ratio < -reward_change_threshold and
                  correct_action_rate_change_ratio < -correct_action_rate_change_threshold):
                # If performance of episode declines, minor increase epsilon decay rate
                self.epsilon_decay *= 1.01
        
        self.prev_performance = performance_metrics
        self.epsilon_decay = max(self.epsilon_decay, self.epsilon_min)

    def calculate_correct_action_rate(self, total_correct_actions, total_actions):
        if total_actions == 0:
            return 0
        return (total_correct_actions / total_actions) * 100


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