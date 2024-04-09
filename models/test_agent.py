import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from env.TradingEnv import *
from models.DQNAgent import *
from preprocessing.preprocess import create_ticker_dict

df = pd.read_csv("models\data\done_data_indicators.csv")
tickers = df['code'].unique()
df_test = df[df['date'] > '2022-12-31']

# Preprocess
df_test = create_ticker_dict(df_test)

env = MultiTickerStockTradingEnv(df_test, tickers, window_size=10)
state_size = env.observation_space[0] * env.observation_space[1] * env.observation_space[2]
action_size = env.action_space
agent = DQNAgent(state_size, action_size)

agent.load_agent('models\\trained_models\model.weights.h5', 'models\\trained_models\\agent_state.pkl')

print(env.max_steps)
rewards = []
profits = []

EPISODES = 10
for episode in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    total_profit = 0
    episode_balances = []
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward[0]
        total_profit += env.balance - env.initial_balance
        episode_balances.append(env.balance)
        state = next_state
    rewards.append(total_reward)
    profits.append(total_profit)
    plt.plot(episode_balances, label=f'Episode {episode+1}')

plt.title('Balance During Testing')
plt.xlabel('Timestep')
plt.ylabel('Balance')
plt.legend()


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(rewards, label='Total Reward')
plt.title('Total Reward During Testing ')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(profits, label='Total Profit')
plt.title('Total Profit During Testing')
plt.xlabel('Episode')
plt.ylabel('Total Profit')
plt.legend()

plt.tight_layout()
plt.show()

