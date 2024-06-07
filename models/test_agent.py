import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from env.TradingEnv import *
from models.DQNAgent import *

from dotenv import load_dotenv
import os
from pymongo import MongoClient


DBUSERNAME = os.environ.get("DB_USERNAME")
DBPASSSWORD = os.environ.get("DB_PASSWORD")
client = MongoClient(f"mongodb+srv://nguyen7obu:iwcwLSDyE0DF22lo@clusterthesis.keduavv.mongodb.net/")
db = client["thesis"]
collection = db["dailyRawData"]

query = {'code': 'VCB', 'date': {'$gte': "2022-12-31", '$lt': "2024-01-01" }}
cursor = collection.find(query)
df = pd.DataFrame(list(cursor))
# df = create_ticker_dict(df)

env = SingleTickerStockTradingEnv(df, window_size=45, initial_balance=7500)
state_size = env.observation_space[0] * env.observation_space[1]
action_size = env.action_space
agent = DQNAgent(state_size, action_size)

model_weights_path = f'models/trained_models/VCB_model_weights.pth'
agent.load_agent(model_weights_path)

state = env.reset()
state = np.reshape(state, [1, state_size])

actions = []
done = False

agent.epsilon = 0

while not done:
    action = agent.act(state, env.shares_held)
    actions.append(action)
    next_state, reward, done, _ = env.step([action])
    state = np.reshape(next_state, [1, state_size])

print(action)