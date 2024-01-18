from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import json
import logging
import requests
from kafka import KafkaProducer
import time
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os

default_args = {
    'owner': 'che',
    'start_date': datetime(2023, 11, 8)
}

TOPIC_NAME = 'raw_realtime'
load_dotenv()
DBUSERNAME = os.environ.get("DB_USERNAME")
DBPASSSWORD = os.environ.get("DB_PASSWORD")

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
    date = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

    producer = KafkaProducer(bootstrap_servers = ['kafka:9092'])
    
    try:
        res = fetch_data(date)
        # res = fetch_data(date)
        res = json.dumps(res).encode('utf-8')
        producer.send('raw_daily', res)
        logging.info('Send to Kafka')
    except Exception as e:
        logging.error(f'An error occured: {e}')


with DAG('daily_dag',
         default_args= default_args,
        #  schedule_interval='0 1 * * 2-6',
         catchup= False) as dag:

    daily_streaming_task = PythonOperator(
        task_id = 'daily_streaming_task',
        python_callable= daily_task
    )

with DAG('realtime_dag',
         default_args= default_args,
         schedule_interval='* 2-7 * * 1-5',
         catchup= False) as dag:
    
    realtime_streaming_task = PythonOperator(
        task_id = 'realtime_streaming_task',
        python_callable= realtime_task
    )