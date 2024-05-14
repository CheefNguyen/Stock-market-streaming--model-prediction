import numpy as np
import random
import pandas as pd
import os

from env.TradingEnv import *
from models.DQNAgent import *
from preprocessing.preprocess import add_technical_indicators, create_ticker_dict

import tensorflow as tf
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("models\data\done_data_indicators.csv")
tickers = df['code'].unique()
df_train = df[df['date'] <= '2023-05-31'] # ~0.8 
df_train = create_ticker_dict(df_train)
# add_technical_indicators(df_train)

print(len(df_train['ACB']))

env = MultiTickerStockTradingEnv(df_train, tickers, window_size=10)
state_size = env.observation_space[0] * env.observation_space[1] * env.observation_space[2]
action_size = env.action_space
agent = DQNAgent(state_size, action_size)

# Parameters
batch_size = 32
EPISODES = 100
target_update_frequency = 10

state = env.reset()
state = np.reshape(state, [1, state_size])

if os.path.exists('models\\trained_models\model_weights.pth') and os.path.exists('models\\trained_models\\agent_state.pkl'):
    agent.load_agent('models\\trained_models\model_weights.pth', 'models\\trained_models\\agent_state.pkl')

# Initialize TensorBoard callback
now = datetime.now().strftime("%Y-%m-%d")
log_dir = f"models\logs\{now}"
if not os.path.exists(log_dir):
     os.makedirs(log_dir)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training loop
for e in range(EPISODES):
    state = env.reset()
    # init first state of Q-table
    state = np.reshape(state, [1, state_size])
    episode_reward  = 0
    episode_profit = 0
    step = 0
    done = False
    prev_balance = 10000
    episode_loss = 0
    while not done:
        action = agent.act(state)
        next_observation, rewards, done, _ = env.step(action)
        reward = rewards[0]  # As we are using a single environment, we consider the first reward
        next_state = np.reshape(next_observation, [1, state_size])
        agent.remember(state, action, reward, next_state, done)

        # episode_balances.append()
        episode_reward += reward

        episode_profit += env.balance - prev_balance
        prev_balance = env.balance

        state = next_state
        if done:
            break
        if len(agent.memory) > batch_size:
            batch_loss = agent.replay(batch_size)
            episode_loss += batch_loss

        if step % target_update_frequency == 0:
            agent.update_target_model()
        
        # with tf.summary.create_file_writer(log_dir).as_default():   
        
        step += 1

    avg_loss  = episode_loss/step
    agent.save_agent('models\\trained_models\model_weights.pth', 'models\\trained_models\\agent_state.pkl')

    with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('Episode Loss', avg_loss, step=e)
            tf.summary.scalar('total_reward', episode_reward, step=e)
            tf.summary.scalar('epsilon', agent.epsilon, step=e)
            tf.summary.scalar('epsilon decay', agent.epsilon_decay, step=e)
            tf.summary.scalar('Episode Balance', env.balance, step=e)
            tf.summary.scalar('Episode Profit', episode_profit, step=e)
            
    print(f"episode: {e+1}/{EPISODES}, avg_loss: {avg_loss}, e: {agent.epsilon}")

    #Update epsilon and epsilon decay for next episode
    agent.update_epsilon()
    agent.performance_update_epsilon_decay([episode_reward, episode_profit])
    