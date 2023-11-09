from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import requests
from kafka import KafkaProducer
import json
import time
import logging

default_args = {
    'owner': 'che',
    'start_date': datetime(2023, 11, 8, 8, 00)
}

def fetch_data(date):
    floor = 'HOSE'
    API_VNDIRECT = f'https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=date:gte:{date}~date:lte:{date}~floor:{floor}&size=99990&page=1'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(API_VNDIRECT,verify=True, headers=headers)
    res = response.json()['data']

    return res

def realtime_task():
    today = datetime.today().strftime('%Y-%m-%d')

    producer = KafkaProducer(bootstrap_servers = 'localhost:9092',
                            max_block_ms = 5000)

    while True:
        try:
            res = fetch_data(today)
            producer.send('raw_data', json.dumps(res).encode('utf-8'))
            logging.info('Send to Kafka')
        except Exception as e:
            logging.error(f'An error occured: {e}')
            continue
        time.sleep(60)

def daily_task():
    date = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

    producer = KafkaProducer(bootstrap_servers = 'localhost:9092',
                            max_block_ms = 5000)
    
    try:
        res = fetch_data(date)
        producer.send('raw_data', json.dumps(res).encode('utf-8'))
        logging.info('Send to Kafka')
    except Exception as e:
        logging.error(f'An error occured: {e}')


with DAG('user_automation',
         default_args= default_args,
         schedule='@daily',
         catchup= False) as dag:

    realtime_streaming_task = PythonOperator(
        task_id = 'realtime_streaming_task',
        python_callable= realtime_task
    )

    daily_streaming_task = PythonOperator(
        task_id = 'daily_streaming_task',
        python_callable= daily_task
    )