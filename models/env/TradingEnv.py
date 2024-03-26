import gym
from gym import spaces
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class StockTradingEnv(gym.Env):
    metadata = {"render.modes":["human"]}

    def __init__(self, df, scaler=None) -> None:
        super(StockTradingEnv, self).__init__()
        self.df = df
        # 3 Actions: Buy - Sell - Hold
        self.action_space = spaces.Box(low=np.array([0,0]), high=np.array([3,1]))
        # Contain last 500 ticks 
        self.observation_space = spaces.Box(low=0, high=1, shape=(501, 501), dtype=np.float16)
        self.scaler = scaler

####

# Import the necessary modules

# Define a custom environment class that inherits from gym.Env
class StockTradingEnv(gym.Env):
    # Initialize the environment with some parameters
    def __init__(self, data, initial_balance, commission, max_steps):
        # data: a pandas dataframe that contains the stock price data
        # initial_balance: the initial amount of money the agent has
        # commission: the percentage of commission fee for each trade
        # max_steps: the maximum number of steps the agent can take in an episode

        # Store the parameters
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps

        # Define the action space as a discrete space of three actions: 0 (hold), 1 (buy), and 2 (sell)
        self.action_space = spaces.Discrete(3)

        # Define the observation space as a box space of four features: balance, shares, price, and step
        # The balance and shares are normalized by the initial balance and the maximum number of shares that can be bought with the initial balance
        # The price is normalized by the maximum price in the data
        # The step is normalized by the max_steps
        self.observation_space = spaces.Box(
            low = np.array([0, 0, 0, 0]), # the lower bound for each feature
            high = np.array([1, 1, 1, 1]), # the upper bound for each feature
            dtype = np.float32 # the data type for each feature
        )

        # Set the initial state
        self.reset()

    # Define the reset method that resets the environment to the initial state
    def reset(self):
        # Set the initial balance, shares, price, and step
        self.balance = self.initial_balance
        self.shares = 0
        self.price = self.data.iloc[0]['Close']
        self.step = 0

        # Return the initial observation as a numpy array
        return np.array([self.balance / self.initial_balance, self.shares / (self.initial_balance / self.price), self.price / self.data['Close'].max(), self.step / self.max_steps])

    # Define the step method that takes an action and returns the next observation, reward, done, and info
    def step(self, action):
        # Check if the action is valid
        assert self.action_space.contains(action)

        # Increment the step
        self.step += 1

        # Get the next price from the data
        self.price = self.data.iloc[self.step]['Close']

        # Initialize the reward, done, and info
        reward = 0
        done = False
        info = {}

        # Perform the action
        if action == 0: # hold
            pass
        elif action == 1: # buy
            # Calculate the number of shares that can be bought with the current balance
            shares = self.balance // (self.price * (1 + self.commission))

            # Update the balance and the shares
            self.balance -= shares * self.price * (1 + self.commission)
            self.shares += shares
        elif action == 2: # sell
            # Calculate the amount of money that can be earned by selling the current shares
            amount = self.shares * self.price * (1 - self.commission)

            # Update the balance and the shares
            self.balance += amount
            self.shares = 0

        # Calculate the total value of the portfolio (balance + shares * price)
        portfolio_value = self.balance + self.shares * self.price

        # Calculate the reward as the change in the portfolio value
        reward = portfolio_value - self.initial_balance

        # Check if the episode is done (the step reaches the max_steps or the portfolio value is zero or negative)
        if self.step == self.max_steps or portfolio_value <= 0:
            done = True

        # Return the next observation, reward, done, and info as a tuple
        return (np.array([self.balance / self.initial_balance, self.shares / (self.initial_balance / self.price), self.price / self.data['Close'].max(), self.step / self.max_steps]), reward, done, info)
