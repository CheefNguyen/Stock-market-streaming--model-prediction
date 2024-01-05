from kafka import KafkaConsumer
import json
import pymongo
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os
import requests

TOPIC_NAME = 'raw_daily'
load_dotenv()
DBUSERNAME = os.environ.get("DB_USERNAME")
DBPASSSWORD = os.environ.get("DB_PASSWORD")

cluster = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")
# cluster = MongoClient("mongodb://localhost:27017")
db = cluster["thesis"]
collection = db["rawDailyData"]

floor = 'HOSE'
# API_VNDIRECT = f'https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=date:gte:2021-01-01~date:lte:2021-06-30~floor:HOSE&size=99990&page=1'
API_VNDIRECT = f'https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=date:gte:2021-07-01~date:lte:2021-12-31~floor:HOSE&size=99990&page=1'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

response = requests.get(API_VNDIRECT,verify=True, headers=headers)
res = response.json()['data']

collection.insert_many(res)