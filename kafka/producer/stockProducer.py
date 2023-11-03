# from airflow import DAG
# from airflow.operators.python import PythonOperator
from datetime import datetime
import json
from kafka import KafkaProducer
import time
import logging
import requests

def fetch_data(date):
    # date_min = "2023-10-03"
    # date_max = "2023-10-03"
    # today = datetime.today().strftime('%Y-%m-%d')
    # prev_day = datetime.today().strftime('%Y-%m-%d')

    floor = 'HOSE'
    API_VNDIRECT = f'https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=date:gte:{date}~date:lte:{date}~floor:{floor}&size=99990&page=1'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(API_VNDIRECT,verify=True, headers=headers)
    res = response.json()['data']

    return res

producer = KafkaProducer(bootstrap_servers = 'localhost:9092',
                            max_block_ms = 5000)
today = datetime.today().strftime('%Y-%m-%d')
while True:
    try:
        res = fetch_data(today)
        producer.send('raw_data', json.dumps(res).encode('utf-8'))
        print('Send')
    except Exception as e:
        logging.error(e)
        continue
    time.sleep(60)