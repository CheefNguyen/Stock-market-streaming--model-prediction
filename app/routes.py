from flask import Flask, request, jsonify, render_template
import numpy as np
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import requests
from datetime import datetime
import pandas as pd
import json

from models.env.TradingEnv import *
from models.models.DQNAgent import *
from models.preprocessing.preprocess import *

app = Flask(__name__, static_folder="static")

# MongoDB connection
load_dotenv()
DBUSERNAME = os.environ.get("DB_USERNAME")
DBPASSSWORD = os.environ.get("DB_PASSWORD")
client = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")
db = client["thesis"]
collection = db["dailyRawData"]

@app.route('/')
def index():
    return  render_template('index.html')

@app.route('/get_realtime_stock_data', methods=['GET'])
def get_stock_data():
    global raw_data
    code = request.args.get('code')
    # today = "2024-05-17"
    today = datetime.today().strftime('%Y-%m-%d')
    API_VNDIRECT = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{code}~date:gte:2024-01-01~date:lte:{today}&size=9990&page=1"
    # API_VNDIRECT = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{code}~date:gte:2022-12-31~date:lte:2024-01-01&size=9990&page=1"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(API_VNDIRECT,verify=True, headers=headers)
    raw_data = response.json()['data']
    return jsonify(raw_data)

@app.route('/get_action_data', methods=['GET'])
def predict():
    global raw_data
    code = request.args.get('code')

    query = {'code': code, 'date': {'$gte': "2024-01-01"}}
    # query = {'code': code, 'date': {'$gte': "2022-12-31", '$lt': "2024-01-01" }} #test
    cursor = collection.find(query)
    df = pd.DataFrame(list(cursor))
    df = df.sort_values('date')
    print(df)

    env = SingleTickerStockTradingEnv(df, window_size=25, initial_balance=100)
    state_size = env.observation_space[0] * env.observation_space[1]
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)

    model_weights_path = f'models/trained_models/{code}_model_weights.pth'
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

    dates = df['date'].iloc[env.window_size:].tolist()
    prices = df['close'].iloc[env.window_size:].tolist()
    
    response_data = {
        'dates': [str(date) for date in dates],  # Convert dates to string
        'actions': [int(action) for action in actions],  # Convert actions to int
        'prices': [float(price) for price in prices]  # Convert prices to float
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)