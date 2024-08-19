from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import json
import logging
import requests
from kafka import KafkaProducer
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os
import pandas as pd
from ta.trend import EMAIndicator, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

default_args = {
    'owner': 'che',
    'start_date': datetime(2023, 11, 8)
}

TOPIC_NAME = 'raw_realtime'
load_dotenv()
DBUSERNAME = 'nguyen7obu'
DBPASSSWORD = 'iwcwLSDyE0DF22lo'
codeList = ["VCB", "MBB", "BID", "EIB"]

def fetch_data(date):
    floor = 'HOSE'
    API_VNDIRECT = f'https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=date:gte:{date}~date:lte:{date}~floor:{floor}&size=99990&page=1'
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

def realtime_task():
    today = datetime.today().strftime('%Y-%m-%d')
    timestamp = datetime.now()

    producer = KafkaProducer(bootstrap_servers = ['kafka:9092'],
                            max_block_ms = 5000)
    logging.info('Connection complete')

    try:
        res = fetch_data(today)
        for item in res:
            item.update({"TimeStamp": timestamp})
            producer.send('raw_realtime', json.dumps(item, default=serialize_datetime).encode('utf-8'))
        logging.info('Send to Kafka')

        # cluster = MongoClient("mongodb://localhost:27017")
        cluster = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")

        db = cluster["thesis"]
        collection = db["rawRealtimeData2"]
        collection.insert_many(res)
    except Exception as e:
        logging.error(f'An error occured: {e}')

def daily_task():
    date = datetime.today() - timedelta(days=1)
    date_str = date.strftime('%Y-%m-%d')
    # date_str = "2024-06-21"
    # producer = KafkaProducer(bootstrap_servers = ['kafka:9092'])
    
    try:
        res = fetch_data(date_str)
        filtered_data = [item for item in res if item.get('code') in codeList]
        filtered_df = pd.DataFrame(filtered_data)

        cluster = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")
        db = cluster["thesis"]
        collection = db["dailyRawData"]

        start_date = date - timedelta(days=60)
        start_date_str = start_date.strftime('%Y-%m-%d')
        for code in codeList:
            query = {'code': code, 'date': {'$gte': start_date_str, '$lte': date_str}}
            cursor = collection.find(query, {"_id": 0}).sort('date', 1)
            df = pd.DataFrame(list(cursor))
            temp = filtered_df[filtered_df['code'] == code]
            df = pd.concat([df, temp], ignore_index= True)

            df = add_technical_indicators(df)
            res = df[df['date'] == date_str].to_dict('records')
            if res:
                collection.insert_many(res)

    except Exception as e:
        logging.error(f'An error occured: {e}')


with DAG('daily_dag',
         default_args= default_args,
         schedule_interval='0 1 * * 2-7',
         catchup= False) as dag:

    daily_streaming_task = PythonOperator(
        task_id = 'daily_streaming_task',
        python_callable= daily_task
    )

# with DAG('realtime_dag',
#          default_args= default_args,
#          schedule_interval='* 2-7 * * 1-5',
#          catchup= False) as dag:
    
#     realtime_streaming_task = PythonOperator(
#         task_id = 'realtime_streaming_task',
#         python_callable= realtime_task
#     )