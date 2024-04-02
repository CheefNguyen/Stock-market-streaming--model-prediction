import numpy as np
import pandas as pd
import random
# import gymnasium as gym

from collections import deque
import sys 
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

sys.path.append(os.path.abspath("./models"))
from env.TradingEnv import MultiTickerOHLCEnv
import config as config

# class MyAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size

#         # Khởi tạo replay buffer
#         self.replay_buffer = deque(maxlen=50000)

#         # Khởi tạo tham số của Agent
#         self.gamma = 0.99
#         self.epsilon = 1.0
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.98
#         self.learning_rate = 0.001
#         self.update_targetnn_rate = 10

#         self.main_network = self.get_nn()
#         self.target_network = self.get_nn()

#         # Update weight của mạng target = mạng main
#         self.target_network.set_weights(self.main_network.get_weights())

#     def get_nn(self):
#         model  = Sequential()
#         model.add (Dense(32, activation='relu', input_dim=self.state_size))
#         model.add (Dense(32, activation='relu'))
#         model.add (Dense(self.action_size))
#         model.compile( loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
#         return model

#     def save_experience(self, state, action, reward, next_state, terminal):
#         self.replay_buffer.append((state, action, reward, next_state, terminal))

#     def get_batch_from_buffer(self, batch_size):

#         exp_batch = random.sample(self.replay_buffer, batch_size)
#         state_batch  = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
#         action_batch = np.array([batch[1] for batch in exp_batch])
#         reward_batch = [batch[2] for batch in exp_batch]
#         next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
#         terminal_batch = [batch[4] for batch in exp_batch]
#         return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

#     def train_main_network(self, batch_size):
#         state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

#         # Lấy Q value của state hiện tại
#         q_values = self.main_network.predict(state_batch, verbose=0)

#         # Lấy Max Q values của state S' (State chuyển từ S với action A)
#         next_q_values = self.target_network.predict(next_state_batch, verbose=0)
#         max_next_q = np.amax(next_q_values, axis=1)

#         for i in range(batch_size):
#             new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
#             q_values[i][action_batch[i]] = new_q_values

#         self.main_network.fit(state_batch, q_values, verbose=0)

#     def make_decision(self, state):
#         if random.uniform(0,1) < self.epsilon:
#             return np.random.randint(self.action_size)

#         state = state.reshape((1, self.state_size))
#         q_values = self.main_network.predict(state, verbose=0)
#         return np.argmax(q_values[0])

class MyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
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

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
df = pd.read_csv("models\data\done_data.csv")
tickers = df['code'].unique()
df = {ticker: df[df["code"] == ticker] for ticker in tickers}
env = MultiTickerOHLCEnv(df, tickers)
state = env.reset()

# Định nghĩa state_size và action_size
state_size = env.observation_space[0] * env.observation_space[1] * env.observation_space[2]
action_size = env.action_space

print(f"Action size: {action_size}")

# Định nghĩa tham số khác
episodes = 100
# n_timesteps = 500
batch_size = 32

# Khởi tạo agent
agent = MyAgent(state_size, action_size)
total_time_step = 0

# Training loop
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(env.max_steps):
        action = agent.act(state)
        next_observation, rewards, done, _ = env.step(action)
        reward = rewards[0]  # As we are using a single environment, we consider the first reward
        next_state = np.reshape(next_observation, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Save weights
agent.main_network.save(f"{config.TRAINED_MODEL_DIR}/train_agent.h5")