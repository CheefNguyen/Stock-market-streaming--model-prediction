from flask import Flask, request, jsonify, render_template
import numpy as np
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import requests
from datetime import datetime

from models.env.TradingEnv import *
from models.models.DQNAgent import *

app = Flask(__name__, static_folder="static")

# MongoDB connection
load_dotenv()
DBUSERNAME = os.environ.get("DB_USERNAME")
DBPASSSWORD = os.environ.get("DB_PASSWORD")
client = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")
db = client["thesis"]
collection = db["rawRealtimeData2"]

@app.route('/')
def index():
    return  render_template('index.html')

@app.route('/get_data', methods=['GET'])
def get_data():
    # Query MongoDB to retrieve data
    data = list(collection.find({}))  

    json_data = data

    return jsonify(json_data)

@app.route('/get_realtime_stock_data', methods=['GET'])
def get_stock_data():
    code = request.args.get('code')
    today = "2024-05-17"
    # today = datetime.today().strftime('%Y-%m-%d')
    API_VNDIRECT = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{code}~date:gte:2024-01-01~date:lte:{today}&size=9990&page=1"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(API_VNDIRECT,verify=True, headers=headers)
    data = response.json()['data']
    return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json['data']

        # Make predictions using DQNAgent model
        predictions = agent.test(input_data)

        # Return predictions as JSON response
        return jsonify({'predictions': predictions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)