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
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(API_VNDIRECT,verify=True, headers=headers)
    raw_data = response.json()['data']
    return jsonify(raw_data)

@app.route('/get_action_data', methods=['GET'])
def predict():
    global raw_data
    code = request.args.get('code')

    query = {'code': code, 'date': {'$gte': "2024-01-01"}}
    cursor = collection.find(query)
    df = pd.DataFrame(list(cursor))
    tickers = df['code'].unique()
    df = create_ticker_dict(df)

    env = MultiTickerStockTradingEnv(df, tickers, window_size=30)
    state_size = env.observation_space[0] * env.observation_space[1] * env.observation_space[2]
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size, num_tickers= 1)

    model_weights_path = 'models/trained_models/2024_05_23/model_weights.pth'
    agent_state_path = 'models/trained_models/2024_05_23/agent_state.pkl'
    agent.load_agent(model_weights_path, agent_state_path)
    
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    actions = []
    done = False

    while not done:
        action = agent.act(state)
        actions.append(action)
        next_state, rewards, done, _ = env.step([action])
        state = np.reshape(next_state, [1, state_size])
    
    actions = [int(action) for action in actions]
    dates = [item['date'] for item in raw_data]
    prices = [item['close'] for item in raw_data]
    
    response_data = {
        'dates': dates,
        'actions': actions,
        'prices': prices
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)