from flask import Flask, request, jsonify, render_template
import numpy as np
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import requests
from datetime import datetime
import pandas as pd
import json

from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from models.env.TradingEnv import *
from models.models.DQNAgent import *
from models.preprocessing.preprocess import *

import random

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

def add_indicators(df):
    df['macd'] = MACD(df['close']).macd()
    df['MACD_Signal'] = MACD(df['close']).macd_signal()
    df['MACD_Histogram'] = MACD(df['close']).macd_diff()
    
    rsi = RSIIndicator(df['close'])
    df['rsi'] = rsi.rsi()
    
    cci = CCIIndicator(df['high'], df['low'], df['close'])
    df['cci'] = cci.cci()
    
    adx = ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    # df['plus_di'] = adx.adx_pos()
    # df['minus_di'] = adx.adx_neg()
    
    return df

@app.route('/get_action_data', methods=['GET'])
def predict():
    global raw_data
    code = request.args.get('code')
    init_balance = float(request.args.get('balance'))
    print(init_balance)

    query = {'code': code, 'date': {'$gte': "2024-01-01"}}

    # query = {'code': code, 'date': {'$lt': "2022-12-31"}} #train
    # query = {'code': code, 'date': {'$gte': "2022-12-31", '$lt': "2024-01-01" }} #test
    cursor = collection.find(query).sort('date', 1)
    df = pd.DataFrame(list(cursor))
    temp_df = pd.DataFrame(raw_data)
    if df.iloc[-1]['date'] != temp_df.iloc[0]['date'] and df.iloc[-1]['date'] > "2024-01-01":
        df = df.append(temp_df.iloc[0], ignore_index = True)
        df = add_indicators(df)
    # print(df)

    model_weights_path = f'models/trained_models/temp5/{code}_model_weights.pth'

    env = SingleTickerStockTradingEnv(df, window_size=25)
    state_size = env.observation_space[0] * env.observation_space[1]
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)

    agent.load_agent(model_weights_path)

    done = False

    actions = []
    balances = []
    portfolio = 0

    # for i in range(20):
    #     balances_ = []
    #     done = False
    state = env.reset(initial_balance=init_balance)
    state = np.reshape(state, [1, state_size])
    agent.epsilon = 0
    agent.model.eval()
    
    while not done:
        current_price = env.data.iloc[env.current_step]['close']
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
    
    
    
    dates = df['date'].iloc[env.window_size:].tolist()
    prices = df['close'].iloc[env.window_size:].tolist()
    
    response_data = {
        'dates': [str(date) for date in dates],
        'actions': [int(action) for action in actions],
        'prices': [float(price) for price in prices],
        'portfolio': portfolio,
        'balance' : balances
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)