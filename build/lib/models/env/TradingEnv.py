import numpy as np
import torch

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
        self.action_space = 3
        self.observation_space = (self.num_tickers, window_size, 8)  # OHLC and 4 Indicators

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.current_step = self.window_size
        return self._get_observation()

    def step(self, actions):
        rewards = []
        for i, ticker in enumerate(self.tickers):
            current_data = self.data[ticker].iloc[self.current_step]

            # Take action
            action = actions[i]
            reward = self._take_action(action, ticker, current_data)
            rewards.append(reward)

        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_observation = self._get_observation()

        info = {
            "correct_actions": self.calculate_correct_actions(actions)  # Example placeholder
        }

        return next_observation, rewards, done, info

    def _take_action(self, action, ticker, current_data):
        reward = 0
        if action == 0:  # Holding
            pass
        elif action == 1:  # Selling
            if self.shares_held[ticker] > 0:
                reward = current_data['close'] * self.shares_held[ticker]
                self.balance += reward
                self.shares_held[ticker] = 0
        elif action == 2:  # Buying
            if self.balance >= current_data['close']:
                self.shares_held[ticker] += 1
                self.balance -= current_data['close']
        return reward

    def _get_observation(self):
        observation = np.zeros((self.num_tickers, self.window_size, 8))
        for i, ticker in enumerate(self.tickers):
            data_slice = self.data[ticker].iloc[self.current_step - self.window_size:self.current_step]

            # Assign OHLC to observation array
            observation[i, :, :4] = torch.tensor(data_slice[['open', 'high', 'low', 'close']].values)

            # Assign Indicators to observation array
            macd = torch.nan_to_num(torch.tensor(data_slice['macd'].values), nan=0.0)
            rsi = torch.nan_to_num(torch.tensor(data_slice['rsi'].values), nan=50.0)
            cci = torch.nan_to_num(torch.tensor(data_slice['cci'].values), nan=0.0)
            adx = torch.nan_to_num(torch.tensor(data_slice['adx'].values), nan=0.0)

            observation[i, :, 4] = macd
            observation[i, :, 5] = rsi
            observation[i, :, 6] = cci
            observation[i, :, 7] = adx

        return observation

    def calculate_correct_actions(self, actions):
        correct_actions = 0
        for i, ticker in enumerate(self.tickers):
            current_data = self.data[ticker].iloc[self.current_step]
            if actions[i] == 2 and current_data['close'] < current_data['open']:  # Buy low
                correct_actions += 1
            if actions[i] == 1 and current_data['close'] > current_data['open']:  # Sell high
                correct_actions += 1
        return correct_actions
    
    def calculate_profit(self):
        total_value = self.balance
        for ticker in self.tickers:
            current_price = self.data[ticker].iloc[self.current_step]['close']
            total_value += self.shares_held[ticker] * current_price
        profit = total_value - self.initial_balance
        return profit