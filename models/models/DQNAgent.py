import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import pickle
from functools import lru_cache

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_update_frequency = 10  # Update target network every 10 episodes

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate)) 
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    # @lru_cache
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            states.append(state)
            targets.append(target_f)

        # Combine states and targets into arrays for efficient training
        states = np.vstack(states)
        targets = np.vstack(targets)

        # Train the model using the entire minibatch data
        self.model.fit(states, targets, epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def save_model_weights(self, filename):
        self.model.save_weights(filename)

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
        self.model.load_weights(filename)

    def load_agent_state(self, filename):
        with open(filename, 'rb') as f:
            self.memory, self.gamma, self.epsilon = pickle.load(f)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay