import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class MultiTickerOHLCEnv(gym.Env):
    def __init__(self, df, tickers, window_size=10, initial_balance=10000):
        super(MultiTickerOHLCEnv, self).__init__()
        self.df = df
        self.tickers = tickers
        self.num_tickers = len(tickers)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = {ticker: 0 for ticker in tickers}
        self.current_step = self.window_size
        self.max_steps = min(len(df[ticker]) for ticker in tickers) - 1

        # Action space: 0 for selling, 1 for holding, 2 for buying for each ticker
        self.action_space = spaces.MultiDiscrete([3] * self.num_tickers)

        # Observation space: OHLC data for each ticker
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_tickers, window_size, 4), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.current_step = self.window_size
        return self._get_observation()

    def step(self, actions):
        # assert len(actions) == self.num_tickers, f"Invalid number of actions: {len(actions)}, expected {self.num_tickers}"

        rewards = 0
        for i, ticker in enumerate(self.tickers):
            current_data = self.df[ticker].iloc[self.current_step]

            # Take action
            action_index = actions[i]
            reward = self._take_action(action_index, ticker, current_data)

            rewards += reward

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
                reward = current_data['Close'] * self.shares_held[ticker]
                self.balance += reward
                self.shares_held[ticker] = 0
        elif action == 1:  # Holding
            pass  # Do nothing
        elif action == 2:  # Buying
            if self.balance >= current_data['Close']:
                self.shares_held[ticker] += 1
                self.balance -= current_data['Close']

        return reward

    def _get_observation(self):
        observation = np.zeros((self.num_tickers, self.window_size, 4))
        for i, ticker in enumerate(self.tickers):
            data_slice = self.df[ticker].iloc[self.current_step - self.window_size:self.current_step]

            observation[i, :, :4] = data_slice[['open', 'high', 'low', 'close']].values
        return observation

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}")