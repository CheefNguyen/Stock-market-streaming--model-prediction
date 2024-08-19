import requests
from datetime import datetime
import pandas as pd

from models.env.TradingEnv import *
from models.models.DQNAgent import *
from models.preprocessing.preprocess import *

from pymongo import MongoClient
from dotenv import load_dotenv
import os

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
from ta.trend import ADXIndicator

from ta.trend import EMAIndicator, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# code = "TCB"
# today = "2024-05-17"
# # today = datetime.today().strftime('%Y-%m-%d')
# # API_VNDIRECT = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{code}~date:gte:2024-01-01~date:lte:{today}&size=9990&page=1"
# API_VNDIRECT = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:VCB~date:gte:2015-01-01~date:lte:2024-08-17&size=99900&page=1"
# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# response = requests.get(API_VNDIRECT,verify=True, headers=headers)
# raw_data = response.json()['data']

# df = pd.DataFrame(raw_data)

# df = df.sort_values(by="date", ascending=True)

# macd = MACD(df['close']).macd()
# macd_signal = MACD(df['close']).macd_signal()

# # Calculate RSI
# rsi = RSIIndicator(df['close']).rsi()

# # Calculate CCI
# cci = CCIIndicator(df['high'], df['low'], df['close']).cci()

# # Calculate ADX
# adx = ADXIndicator(df['high'], df['low'], df['close']).adx()

# # Add indicators to DataFrame
# df['macd'] = macd
# df['MACD_Signal'] = macd_signal

# df['rsi'] = rsi
# df['cci'] = cci
# df['adx'] = adx

# # load_dotenv()
# # DBUSERNAME = os.environ.get("DB_USERNAME")
# # DBPASSSWORD = os.environ.get("DB_PASSWORD")
# # cluster = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")


# # data_dict = df.to_dict(orient='records')

# # db = cluster["thesis"]
# # collection = db["dailyRawData"]
# # collection.insert_many(data_dict)

# # print(df[df['date'] == "2024-05-17"].to_dict('records'))

TOPIC_NAME = 'raw_realtime'
load_dotenv()
DBUSERNAME = 'nguyen7obu'
DBPASSSWORD = 'iwcwLSDyE0DF22lo'
codeList = ["VCB", "MBB", "BID", "EIB"]

def fetch_data(date):
    floor = 'HOSE'
    API_VNDIRECT = f'https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=date:gte:2024-07-19~date:lte:2024-08-18~floor:{floor}&size=99990&page=1'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(API_VNDIRECT,verify=True, headers=headers)
    res = response.json()['data']
    return res

def serialize_datetime(obj): 
    if isinstance(obj, datetime): 
        return obj.isoformat()

def add_technical_indicators(df):
    # Calculate 10-day and 20-day EMAs
    ema_10 = EMAIndicator(df['close'], window=10)
    ema_20 = EMAIndicator(df['close'], window=20)
    df['ema_10'] = ema_10.ema_indicator()
    df['ema_20'] = ema_20.ema_indicator()

    # Calculate RSI
    rsi = RSIIndicator(df['close']).rsi()

    # Calculate CCI
    cci = CCIIndicator(df['high'], df['low'], df['close']).cci()

    # Calculate ADX
    adx = ADXIndicator(df['high'], df['low'], df['close']).adx()

    # Add indicators to DataFrame
    df['macd'] = df['ema_10'] - df['ema_20']
    df['MACD_Signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = adx
    df = df.drop(['ema_10', 'ema_20'], axis=1)
    return df

def daily_task():
    # date_str = "2024-06-21"
    # producer = KafkaProducer(bootstrap_servers = ['kafka:9092'])
    
    try:
        res = fetch_data("date_str")
        filtered_data = [item for item in res if item.get('code') in codeList]
        filtered_df = pd.DataFrame(filtered_data)

        cluster = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")
        db = cluster["thesis"]
        collection = db["dailyRawData"]

        for code in codeList:
            query = {'code': code, 'date': {'$gte': "2024-01-01", '$lte': "2024-07-18"}}
            cursor = collection.find(query, {"_id": 0}).sort('date', 1)
            df = pd.DataFrame(list(cursor))
            temp = filtered_df[filtered_df['code'] == code]
            df = pd.concat([df, temp], ignore_index= True)

            df = add_technical_indicators(df)
            res = df[df['date'] >= "2024-07-18"].to_dict('records')
            if res:
                collection.insert_many(res)
                # print(res)

    except Exception as e:
        print(f'An error occured: {e}')
daily_task()
