import numpy as np
import random
import pandas as pd

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
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

class MultiTickerStockTradingEnv:
    def __init__(self, data, tickers, window_size=10, initial_balance=10000):
        self.data = data
        self.tickers = tickers
        self.num_tickers = len(tickers)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = {ticker: 0 for ticker in tickers}
        self.current_step = self.window_size
        self.max_steps = min(len(data[ticker]) for ticker in tickers) - 1
        self.action_space = np.prod([3] * self.num_tickers)
        self.observation_space = (self.num_tickers, window_size, 9)

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.current_step = self.window_size
        return self._get_observation()

    def step(self, actions):
        # assert actions == self.num_tickers, f"Invalid number of actions: {actions}, expected {self.num_tickers}"

        rewards = []
        for i, ticker in enumerate(self.tickers):
            current_data = self.data[ticker].iloc[self.current_step]

            # Take action
            action_index = actions
            reward = self._take_action(action_index, ticker, current_data)
            rewards.append(reward)

        # Move to the next time step
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        # Get the next observation
        next_observation = self._get_observation()

        return next_observation, rewards, done, {}

    def _take_action(self, action, ticker, current_data):
        reward = 0
        if action == 0:  # Selling
            if self.shares_held[ticker] > 0:
                reward = current_data['close'] * self.shares_held[ticker]
                self.balance += reward
                self.shares_held[ticker] = 0
        elif action == 1:  # Holding
            pass  # Do nothing
        elif action == 2:  # Buying
            if self.balance >= current_data['close']:
                self.shares_held[ticker] += 1
                self.balance -= current_data['close']

        return reward

    def _get_observation(self):
        observation = np.zeros((self.num_tickers, self.window_size, 9))
        for i, ticker in enumerate(self.tickers):
            data_slice = self.data[ticker].iloc[self.current_step - self.window_size:self.current_step]

            # Calculate technical indicators

            # Assign OHLC and indicators to observation array
            observation[i, :, :4] = data_slice[['open', 'high', 'low', 'close']].values

        return observation

# Initialize environment and agent
# tickers = ['AAPL', 'GOOGL', 'MSFT']
# data = {ticker: pd.DataFrame({
#     'Open': np.random.rand(100),
#     'High': np.random.rand(100),
#     'Low': np.random.rand(100),
#     'Close': np.random.rand(100)
# }) for ticker in tickers}

df = pd.read_csv("models\data\done_data.csv")
tickers = df['code'].unique()
df = {ticker: df[df["code"] == ticker] for ticker in tickers}

env = MultiTickerStockTradingEnv(df, tickers)
state_size = env.observation_space[0] * env.observation_space[1] * env.observation_space[2]
action_size = env.action_space
agent = DQNAgent(state_size, action_size)

# Parameters
batch_size = 32
EPISODES = 10

for _ in range(5):
    rand = np.random.randint(3)
    print(f'Random action {rand}')
    print(env.step(rand))

# # Training loop
# for e in range(EPISODES):
#     state = env.reset()
#     state = np.reshape(state, [1, state_size])
#     for time in range(env.max_steps):
#         action = agent.act(state)
#         next_observation, rewards, done, _ = env.step(action)
#         reward = rewards[0]  # As we are using a single environment, we consider the first reward
#         next_state = np.reshape(next_observation, [1, state_size])
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
#             break
#         if len(agent.memory) > batch_size:
#             agent.replay(batch_size)