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

df = pd.read_csv("models\data\done_data.csv")
tickers = df['code'].unique()
df_train = df[df['date'] <= '2022-12-31']
df_train = create_ticker_dict(df_train)
add_technical_indicators(df_train)

env = MultiTickerStockTradingEnv(df_train, tickers, window_size=10)
state_size = env.observation_space[0] * env.observation_space[1] * env.observation_space[2]
action_size = env.action_space
agent = DQNAgent(state_size, action_size)

# Parameters
batch_size = 64
EPISODES = 100
target_update_frequency = 10

state = env.reset()
print(state.size)
state = np.reshape(state, [1, state_size])

if os.path.exists('models\\trained_models\model.weights.h5') and os.path.exists('models\\trained_models\\agent_state.pkl'):
    agent.load_agent('models\\trained_models\model.weights.h5', 'models\\trained_models\\agent_state.pkl')

# Initialize TensorBoard callback
log_dir = "models\logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training loop
for e in range(EPISODES):
    state = env.reset()
    # init first state of Q-table
    state = np.reshape(state, [1, state_size])
    episode_reward  = 0
    step = 0
    done = False
    while not done:
        step += 1
        print(step)
        action = agent.act(state)
        next_observation, rewards, done, _ = env.step(action)
        reward = rewards[0]  # As we are using a single environment, we consider the first reward
        next_state = np.reshape(next_observation, [1, state_size])
        agent.remember(state, action, reward, next_state, done)

        # episode_balances.append()
        episode_reward += reward

        state = next_state
        if done:
            print(f"episode: {e+1}/{EPISODES}, score: {step}, e: {agent.epsilon}".format(e, EPISODES, step, agent.epsilon))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            agent.update_epsilon()

        if step % target_update_frequency == 0:
            agent.update_target_model()
        
        if step % 100 == 0 and step > batch_size:
            agent.save_agent('models\\trained_models\model.weights.h5', 'models\\trained_models\\agent_state.pkl')

    with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('total_reward', episode_reward, step=e)
            tf.summary.scalar('epsilon', agent.epsilon, step=e)
            tf.summary.scalar('Episode Balance', env.balance, step=e)
            
agent.save_agent('models\\trained_models\model.weights.h5', 'models\\trained_models\\agent_state.pkl')
