import numpy as np
import torch

class SingleTickerStockTradingEnv:
    def __init__(self, data, ticker, window_size=10, initial_balance=10000):
        self.data = data
        self.ticker = ticker
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        self.max_steps = len(self.data) - 1
        self.action_space = 3
        self.observation_space = (window_size, 8)  # OHLC and 4 Indicators

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        return self._get_observation()

    def step(self, action):
        current_data = self.data.iloc[self.current_step]

        reward = self._take_action(action, current_data)

        done = self.current_step >= self.max_steps
        next_observation = self._get_observation()

        info = {
            "correct_action": self.calculate_correct_action(action, current_data)
        }

        return next_observation, reward, done, info

    def _take_action(self, action, ticker, current_data):
        reward = 0
        if action == 0:  # Holding
            pass
        elif action == 1:  # Selling
            if self.shares_held > 0:
                reward = current_data['close'] * self.shares_held
                self.balance += reward
                self.shares_held = 0
        elif action == 2:  # Buying
            if self.balance >= current_data['close']:
                self.shares_held += 1
                self.balance -= current_data['close']
        return reward

    def _get_observation(self):
        data_slice = self.data.iloc[self.current_step - self.window_size:self.current_step]
        observation = np.zeros((self.window_size, 8))
        
        # Assign OHLC to observation array
        observation[:, :4] = torch.tensor(data_slice[['open', 'high', 'low', 'close']].values)

        # Assign Indicators to observation array
        macd = torch.nan_to_num(torch.tensor(data_slice['macd'].values), nan=0.0)
        rsi = torch.nan_to_num(torch.tensor(data_slice['rsi'].values), nan=50.0)
        cci = torch.nan_to_num(torch.tensor(data_slice['cci'].values), nan=0.0)
        adx = torch.nan_to_num(torch.tensor(data_slice['adx'].values), nan=0.0)

        observation[:, 4] = macd
        observation[:, 5] = rsi
        observation[:, 6] = cci
        observation[:, 7] = adx

        return observation

    def calculate_correct_action(self, action, current_data):
        if action == 2 and current_data['close'] < current_data['open']:  # Buy low
            return 1
        if action == 1 and current_data['close'] > current_data['open']:  # Sell high
            return 1
        return 0
    
    def calculate_profit(self):
        total_value = self.balance + self.shares_held * self.data.iloc[self.current_step]['close']
        profit = total_value - self.initial_balance
        return profit