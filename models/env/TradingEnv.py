import numpy as np
import torch

class SingleTickerStockTradingEnv:
    def __init__(self, data, window_size=25, initial_balance=500):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        self.max_steps = len(self.data) - 1
        self.action_space = 3
        self.observation_space = (window_size, 8)  # OHLC and 4 Indicators
        self.prev_action = 0

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        return self._get_observation()

    def step(self, action):
        current_data = self.data.iloc[self.current_step]

        reward = self._take_action(action, current_data)

        self.current_step += 1

        done = self.current_step >= self.max_steps
        next_observation = self._get_observation()

        info = {
            "correct_action_reward": self.calculate_correct_action_reward(action, current_data)
        }

        return next_observation, reward, done, info

    def _take_action(self, action, current_data):
        reward = 0

        if action == 0:  # Holding
            reward = 0.1  # Small reward for holding to prevent overtrading
            reward += self.calculate_correct_action_reward(action, current_data)

        elif action == 1:  # Selling
            if self.shares_held > 0:
                sell_value = current_data['close'] * self.shares_held
                self.balance += sell_value
                reward = (sell_value - current_data['close'])  # Reward is profit from selling
                reward += self.calculate_correct_action_reward(action, current_data)
                self.shares_held = 0
            else:
                reward = -20  # Penalty for invalid sell action
        elif action == 2:  # Buying
            if self.balance >= current_data['close']:
                self.shares_held += 1
                self.balance -= current_data['close']
                reward = self.calculate_correct_action_reward(action, current_data)
            else:
                reward = -20  # Penalty for invalid buy action

        # Penalize for consecutive buy/sell actions
        if action == self.prev_action:
            reward -= 10  # Penalty for consecutive buy/sell actions

        self.prev_action = action  # Update the previous action
        
        return reward

    def _get_observation(self):
        data_slice = self.data.iloc[self.current_step - self.window_size:self.current_step]
        observation = np.zeros((self.window_size, 8))
        
        # Assign OHLC to observation array
        observation[:, :4] = data_slice[['open', 'high', 'low', 'close']].values

        # Assign Indicators to observation array
        observation[:, 4] = np.nan_to_num(data_slice['macd'].values, nan=0.0)
        observation[:, 5] = np.nan_to_num(data_slice['rsi'].values, nan=50.0)
        observation[:, 6] = np.nan_to_num(data_slice['cci'].values, nan=0.0)
        observation[:, 7] = np.nan_to_num(data_slice['adx'].values, nan=0.0)

        return observation
    
    def calculate_profit(self):
        total_value = self.balance + self.shares_held * self.data.iloc[self.current_step]['close']
        # Current balance + shares hold current price 
        profit = total_value - self.initial_balance
        return profit
    
    def calculate_correct_action_reward(self, action, current_data):
        correct_action_reward = 0

        if action == 2:  # Buying
            if current_data['macd'] > 0 and current_data['rsi'] < 30 and current_data['cci'] < -100:
                correct_action_reward += 20  # Strong buy signal
            elif current_data['macd'] > 0 and current_data['rsi'] < 50:
                correct_action_reward += 15  # Moderate buy signal
            elif current_data['macd'] > 0 or current_data['rsi'] < 30:
                correct_action_reward += 10  # Weak buy signal

        # Strong sell signals
        if action == 1:  # Selling
            if current_data['macd'] < 0 and current_data['rsi'] > 70 and current_data['cci'] > 100:
                correct_action_reward += 20  # Strong sell signal
            elif current_data['macd'] < 0 and current_data['rsi'] > 50:
                correct_action_reward += 15  # Moderate sell signal
            elif current_data['macd'] < 0 or current_data['rsi'] > 70:
                correct_action_reward += 10  # Weak sell signal

        # Penalize holding during strong signals
        if action == 0:  # Holding
            if current_data['macd'] > 0 and current_data['rsi'] < 30 and current_data['cci'] < -100:
                correct_action_reward -= 10  # Holding during strong buy signal
            elif current_data['macd'] < 0 and current_data['rsi'] > 70 and current_data['cci'] > 100:
                correct_action_reward -= 10  # Holding during strong sell signal
            elif current_data['adx'] < 20:
                correct_action_reward += 5  # Reward for holding during low volatility

        # Penalize general overtrading
        if action != 0:
            correct_action_reward -= 5  # General penalty for overtrading

        return correct_action_reward