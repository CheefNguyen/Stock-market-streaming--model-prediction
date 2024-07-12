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
cursor = collection.find(query).sort('date', 1)
df = pd.DataFrame(list(cursor))
tickers = df['code'].unique()
tickers = ['VCB', 'BID']

envs = {}
agents = {}
state_sizes = {}
# balances = {'VCB' : 2500,
#             'EIB' : 500,
#             'MBB' : 500,
#             'BID' : 2000}

for ticker in tickers:
    df_ticker = df[df['code'] == ticker].reset_index(drop=True)
    df_ticker = df_ticker.sort_values('date')
    env = SingleTickerStockTradingEnv(df_ticker, initial_balance=2500)
    state_size = env.observation_space[0] * env.observation_space[1]
    agent = DQNAgent(state_size, 3)

    envs[ticker] = env
    agents[ticker] = agent
    state_sizes[ticker] = state_size

# Parameters
batch_size = 64
EPISODES = 950
target_update_frequency = 200
replay_freq = 50


# Initialize TensorBoard callback
now = datetime.now().strftime("%Y-%m-%d")


# Training loop
for ticker in tickers:
    balance_range=(500, 5000)
    env = envs[ticker]
    agent = agents[ticker]
    state_size = state_sizes[ticker]

    log_dir = f"models\logs\{now}\{ticker}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    rewards = []
    for e in range(EPISODES):
        # initial_balance = np.random.uniform(balance_range[0], balance_range[1])
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        actions_reward = 0
        total_q_values = 0
        total_actions = 0
        initial_balance = env.initial_balance
        done = False
        step_count = 0
        actions = []
        while not done:
            state = np.reshape(state, [1, state_size])
            action = agent.act(state)
            actions.append(action)
            next_state, reward, done, info = env.step(action)
            step_count += 1

            episode_reward += reward
            actions_reward += info["correct_action_reward"] if "correct_action_reward" in info else 0
            if action != 0:
                total_actions += 1
    
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) >= batch_size and step_count % replay_freq == 0:
                episode_loss = agent.replay(batch_size)
            
            if step_count % target_update_frequency == 0:
                    agent.update_target_model()

            if done:
                break
            # if action != 0:
            #     print(f"Step: {env.current_step}, Action: {action}, Reward: {reward}, Profit: {info['profit']}")


        agent.update_epsilon()

        episode_porfolio = env.calculate_portfolio()
        episode_balance = env.balance
        rewards.append(episode_reward)

        with tf.summary.create_file_writer(log_dir).as_default():
                if e % 5 == 0:
                    tf.summary.scalar('Avg Reward per Episode', np.mean(rewards), step=e)
                    rewards = []
                tf.summary.scalar('epsilon', agent.epsilon, step=e)
                tf.summary.scalar('epsilon decay', agent.epsilon_decay, step=e)
                tf.summary.scalar('Episode Portfolio', episode_porfolio, step=e)
                tf.summary.scalar('Episode Balance', episode_balance, step=e)
                tf.summary.scalar('Episode Loss', episode_loss, step=e)

        print(f"{ticker} _Episode: {e+1}/{EPISODES}, epsilon: {agent.epsilon}, Total actions: {total_actions}, Episode reward: {episode_reward}, %profit: {(episode_porfolio/initial_balance)*100}%, balance: {episode_balance}, initital_balance: {initial_balance}")

        performance_metrics = (episode_porfolio, episode_reward)
        # agent.performance_update_epsilon_decay(performance_metrics)

        agent.save_agent(f'models/trained_models/{ticker}_model_weights.pth')