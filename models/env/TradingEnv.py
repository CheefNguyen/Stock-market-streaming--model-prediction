import numpy as np

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
        self.observation_space = (self.num_tickers, window_size, 8) # OHLC and 4 Indicators

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
        if action == 0:  # Holding
            pass  # Do nothing
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
            observation[i, :, :4] = data_slice[['open', 'high', 'low', 'close']].values

            # Assign Indicators to observation array
            macd = np.nan_to_num(data_slice['macd'].values, nan=0.0)
            rsi = np.nan_to_num(data_slice['rsi'].values, nan=50.0)
            cci = np.nan_to_num(data_slice['cci'].values, nan=0.0)
            adx = np.nan_to_num(data_slice['adx'].values, nan=0.0) 

            observation[i, :, 4] = macd
            observation[i, :, 5] = rsi
            observation[i, :, 6] = cci
            observation[i, :, 7] = adx

        return observation
    