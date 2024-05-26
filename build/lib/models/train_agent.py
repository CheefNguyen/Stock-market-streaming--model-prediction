import numpy as np
import random
import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv

from env.TradingEnv import *
from models.DQNAgent import *
from preprocessing.preprocess import add_technical_indicators, create_ticker_dict

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
print(df.shape)
tickers = df['code'].unique()
df_train = create_ticker_dict(df)

# tickers = df['code'].unique()
# df_train = df[df['date'] <= '2023-05-31'] # ~0.8 
# df_train = create_ticker_dict(df_train)
# add_technical_indicators(df_train)

env = MultiTickerStockTradingEnv(df_train, tickers, window_size=30)
state_size = env.observation_space[0] * env.observation_space[1] * env.observation_space[2]
action_size = env.action_space
agent = DQNAgent(state_size, action_size, num_tickers=env.num_tickers)

# Parameters
batch_size = 32
EPISODES = 1000
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
    episode_reward = 0
    episode_loss = 0
    total_rewards = []
    total_correct_actions = 0
    total_q_values = 0
    total_actions = 0
    initial_balance = env.balance
    done = False
    while not done:
        state = np.reshape(state, [1, state_size])
        action = agent.act(state)
        next_state, rewards, done, info = env.step(action)

        total_rewards.append(sum(rewards))
        total_correct_actions += info["correct_actions"] if "correct_actions" in info else 0
        total_actions += 1

        agent.remember(state, action, sum(rewards), next_state, done)
        state = next_state

        if done:
            break
    
    episode_loss = agent.replay(batch_size)
    if e % target_update_frequency == 0:
        agent.update_target_model()
    agent.update_epsilon()

    episode_profit = env.calculate_profit()

    correct_action_rate = agent.calculate_correct_action_rate(total_correct_actions, total_actions)

    with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('Episode Reward', sum(total_rewards), step=e)
            tf.summary.scalar('epsilon', agent.epsilon, step=e)
            tf.summary.scalar('epsilon decay', agent.epsilon_decay, step=e)
            tf.summary.scalar('Episode Profit', episode_profit, step=e)
            tf.summary.scalar('Episode Loss', episode_loss, step=e)
            
    print(f"episode: {e+1}/{EPISODES}, avg_loss: {episode_loss}, e: {agent.epsilon}")

    #Update epsilon and epsilon decay for next episode

    performance_metrics = (sum(total_rewards),correct_action_rate)
    agent.performance_update_epsilon_decay(performance_metrics)

    agent.save_agent('models\\trained_models\model_weights.pth', 'models\\trained_models\\agent_state.pkl')
    