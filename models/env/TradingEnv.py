import gymnasium as gym
import numpy as np
import pandas as pd

class OHLCEnvironment(gym.Env):
    def __init__(self, df):
        super(OHLCEnvironment, self).__init__()
        self.df = df  # Load your OHLC df here
        self.current_step = 0
        self.max_steps = len(self.df)

        # Define observation space and action space
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        # Execute action (e.g., update portfolio, calculate reward)
        # Update self.current_step
        # Return next observation, reward, done, info
        done = False

        # Example: Assume a simple buy-and-hold strategy
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']

        if action == 0:  # Buy
            reward = next_price - current_price
        elif action == 1:  # Sell
            reward = current_price - next_price
        else:  # Hold
            reward = 0

        self.current_step += 1
        done = self.current_step >= self.max_steps
        pass

    def _get_observation(self):
        # Get OHLC df for the current step
        return np.array(self.df.iloc[self.current_step])
