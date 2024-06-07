import numpy as np
import random
import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv

from env.TradingEnv import *
from models.DQNAgent import *

import tensorflow as tf
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime

load_dotenv()
DBUSERNAME = os.environ.get("DB_USERNAME")
DBPASSSWORD = os.environ.get("DB_PASSWORD")
client = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")
db = client["thesis"]
collection = db["dailyRawData"]

# df = pd.read_csv("models\data\done_data_indicators.csv")

query = {'date': {'$lt': "2022-12-31"}}
cursor = collection.find(query)
df = pd.DataFrame(list(cursor))
tickers = df['code'].unique()
# tickers = ['MBB', 'BID', 'EIB']

envs = {}
agents = {}
state_sizes = {}

for ticker in tickers:
    df_ticker = df[df['code'] == ticker].reset_index(drop=True)
    df_ticker = df_ticker.sort_values('date')
    env = SingleTickerStockTradingEnv(df_ticker)
    state_size = env.observation_space[0] * env.observation_space[1]
    agent = DQNAgent(state_size, 3)

    envs[ticker] = env
    agents[ticker] = agent
    state_sizes[ticker] = state_size

# Parameters
batch_size = 64
EPISODES = 500
target_update_frequency = 10


# Initialize TensorBoard callback
now = datetime.now().strftime("%Y-%m-%d")
log_dir = f"models\logs\{now}"
if not os.path.exists(log_dir):
     os.makedirs(log_dir)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training loop
for ticker in tickers:
    env = envs[ticker]
    agent = agents[ticker]
    state_size = state_sizes[ticker]

    for e in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        actions_reward = 0
        total_q_values = 0
        total_actions = 0
        initial_balance = env.initial_balance
        done = False
        while not done:
            state = np.reshape(state, [1, state_size])
            action = agent.act(state, env.shares_held)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            actions_reward += info["correct_action_reward"] if "correct_action_reward" in info else 0
            if action != 0:
                total_actions += 1
    
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        if len(agent.memory) >= batch_size:
            episode_loss = agent.replay(batch_size)
            if e % target_update_frequency == 0:
                agent.update_target_model()

        agent.update_epsilon()

        episode_profit = env.calculate_profit()

        with tf.summary.create_file_writer(log_dir).as_default():
                tf.summary.scalar('Episode Reward', episode_reward, step=e)
                tf.summary.scalar('epsilon', agent.epsilon, step=e)
                tf.summary.scalar('epsilon decay', agent.epsilon_decay, step=e)
                tf.summary.scalar('Episode Profit', episode_profit, step=e)
                tf.summary.scalar('Episode Loss', episode_loss, step=e)
                
        print(f"{ticker} _Episode: {e+1}/{EPISODES}, epsilon: {agent.epsilon}, Total actions: {total_actions}, action reward: {actions_reward}, %profit: {(episode_profit/initial_balance)*100}%")

        performance_metrics = (episode_profit, episode_reward)
        # agent.performance_update_epsilon_decay(performance_metrics)

        agent.save_agent(f'models/trained_models/{ticker}_model_weights.pth')