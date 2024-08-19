import numpy as np
import pandas as pd
from env.TradingEnv import SingleTickerStockTradingEnv
from models.DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import random
import torch

# Load data from MongoDB or CSV
def load_data_from_mongodb(code):
    from pymongo import MongoClient
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DBUSERNAME = os.environ.get("DB_USERNAME")
    DBPASSSWORD = os.environ.get("DB_PASSWORD")
    client = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")
    db = client["thesis"]
    collection = db["dailyRawData"]

    query = {'code': code, 'date': {'$gte': "2022-12-31", '$lt': "2024-01-01" }} #test
    cursor = collection.find(query)
    df = pd.DataFrame(list(cursor))
    df = df.sort_values('date')
    return df

# Test function for the agent
def test_agent(code, model_weights_path, initial_balance=500, window_size=25, start_date="2024-01-01", end_date="2024-12-31"):
    df = load_data_from_mongodb(code)

    env = SingleTickerStockTradingEnv(df, window_size=window_size, initial_balance=initial_balance)
    state_size = env.observation_space[0] * env.observation_space[1]
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)

    agent.load_agent(model_weights_path)

    state = env.reset()
    state = np.reshape(state, [1, state_size])

    done = False
    agent.epsilon = 0

    actions = []
    rewards = 0
    balances = []

    while not done:
        action = agent.act(state)

        # if action == 1 and env.shares_held == 0:
        #     action = 0
        # if action == 2 and env.balance < current_price:
        #     action = 0

        next_state, reward, done, info = env.step(action)
        balances.append(env.balance)

        actions.append(action)
        state = np.reshape(next_state, [1, state_size])

    portfolio = env.calculate_portfolio()
    dates = df['date'].iloc[env.window_size:]
    plt.figure(figsize=(12, 6))
    plt.plot(dates, balances, label='Balance', color='blue')
    plt.scatter(dates, balances, c=actions, cmap='viridis', label='Actions')
    plt.colorbar(label='Actions (0: Hold, 1: Sell, 2: Buy)')
    plt.title('Balance and Actions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Balance')
    plt.legend()
    plt.show()


    return None

# Example usage
if __name__ == "__main__":
    code = "VCB"
    model_weights_path = f'models/trained_models/temp4/{code}_model_weights.pth'
    initial_balance = 500
    window_size = 25
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    test_agent(code, model_weights_path, initial_balance, window_size, start_date, end_date)