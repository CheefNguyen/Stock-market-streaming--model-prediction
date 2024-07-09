import numpy as np
import torch

class SingleTickerStockTradingEnv:
    def __init__(self, data, window_size=25, initial_balance=5000):
        self.data = data
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.shares_held = 0
        self.current_step = self.window_size
        self.max_steps = len(self.data) - 1
        self.action_space = 3
        self.observation_space = (window_size, 9)  # OHLC and 4 Indicators
        self.last_trade = 0
        self.buy_price = []
        self.closing_price_history = []

    def reset(self):
        self.balance = float(self.initial_balance)
        self.shares_held = 0
        self.current_step = self.window_size
        self.last_trade = 0
        self.buy_price = []
        self.closing_price_history = []
        return self._get_observation()

    def step(self, action):
        done = self.current_step >= self.max_steps

        current_data = self.data.iloc[self.current_step]

        reward = self._take_action(action, current_data)

        self.current_step += 1

        next_observation = self._get_observation()

        info = {
            "correct_action_reward": self.calculate_reward(action, current_data, )
        }

        self.closing_price_history.append(current_data['close'])

        return next_observation, reward, done, info

    def _take_action(self, action, current_data):
        reward = 0

        if action == 0:  # Holding
            reward += self.calculate_reward(action, current_data, )

        elif action == 1:  # Selling
            if self.shares_held > 0:
                sell_value = current_data['close'] * self.shares_held
                if self.balance < current_data['close']:
                    reward += 2
                self.balance += sell_value
                profit = sell_value - sum(self.buy_price)
                reward += self.calculate_reward(action, current_data, profit)
                self.shares_held = 0
                self.buy_price = []
                self.last_trade = self.current_step
            else:
                reward -= 10

        elif action == 2:  # Buying
            if self.balance >= current_data['close']:
                self.shares_held += 1
                self.balance -= current_data['close']
                if len(self.buy_price) > 0 and current_data['close'] < self.buy_price[-1]:
                    reward += 3
                self.last_trade = self.current_step
                self.buy_price.append(current_data['close'])
                reward += self.calculate_reward(action, current_data)
            else:
                reward -= 5

        if self.balance < current_data['close'] and self.shares_held > 0 and action == 2:
            reward -= 1000  # Penalty for not selling to cut losses

        return reward

    def calculate_reward(self, action, current_data, profit=0):
        reward = 0
        
        # Profit-based rewards
        if action == 1 and self.shares_held > 0:  # Selling
            if profit > 0:
                reward += profit * 10  # Positive reward for profitable trades
            
            # MACD
            if current_data['macd'] < current_data['MACD_Signal']:
                reward += 0.5
            elif current_data['macd'] > current_data['MACD_Signal']:
                reward -= 0.5

            # RSI
            if current_data['rsi'] > 55:
                reward += 0.5
            elif current_data['rsi'] < 45:
                reward -= 0.5

            # CCI
            if current_data['cci'] > 50:
                reward += 0.5
            elif current_data['cci'] < -50:
                reward -= 0.5

            # ADX
            if current_data['adx'] > 20:
                reward += 0.5
            elif current_data['cci'] < 20:
                reward -= 0.5
        
        # Technical indicator-based rewards
        if action == 2:  # Buying
            # MACD
            if current_data['macd'] > current_data['MACD_Signal']:
                reward += 0.5
            elif current_data['macd'] < current_data['MACD_Signal']:
                reward -= 0.5
            
            # RSI
            if current_data['rsi'] < 45:
                reward += 0.5
            elif current_data['rsi'] > 55:
                reward -= 0.5
            
            # CCI
            if current_data['cci'] < -50:
                reward += 0.5
            elif current_data['cci'] > 50:
                reward -= 0.5

            # ADX
            if current_data['adx'] > 20:
                reward += 0.5
            elif current_data['cci'] < 20:
                reward -= 0.5

        # Penalize holding during strong signals
        if action == 0:  # Holding
            if (current_data['macd'] > current_data['MACD_Signal'] and 
                current_data['rsi'] < 45 and 
                current_data['cci'] < -50 and current_data['adx'] > 20):
                reward -= 5  # Holding during strong buy signal
            elif (current_data['macd'] < current_data['MACD_Signal'] and 
                  current_data['rsi'] > 55 and 
                  current_data['cci'] > 50 and current_data['adx'] > 20):
                reward -= 5  # Holding during strong sell signal
            elif current_data['adx'] < 20:
                reward += 0.5  # Reward for holding during a weak trend (low ADX)

        if action == 0 and self.current_step - self.last_trade >= 1 and self.current_step - self.last_trade <= 5:
            reward += 1  # Reward holding for a longer period
        elif self.current_step - self.last_trade > 10:
            reward -= 5

        return reward

    def _get_observation(self):
        data_slice = self.data.iloc[self.current_step - self.window_size:self.current_step]
        observation = np.zeros((self.window_size, 9))
        
        # Assign OHLC to observation array
        observation[:, :4] = data_slice[['open', 'high', 'low', 'close']].values

        # Assign Indicators to observation array
        observation[:, 4] = np.nan_to_num(data_slice['macd'].values, nan=0.0)
        observation[:, 5] = np.nan_to_num(data_slice['rsi'].values, nan=50.0)
        observation[:, 6] = np.nan_to_num(data_slice['cci'].values, nan=0.0)
        observation[:, 7] = np.nan_to_num(data_slice['adx'].values, nan=0.0)
        observation[:, 8] = np.nan_to_num(data_slice['MACD_Signal'].values, nan=0.0)

        return observation
    
    def calculate_portfolio(self):
        total_value = self.balance + self.shares_held * self.data.iloc[self.max_steps]['close']
        return total_value