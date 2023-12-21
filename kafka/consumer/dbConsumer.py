from kafka import KafkaConsumer
import json
import pymongo
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os

TOPIC_NAME = 'raw_realtime'
load_dotenv()
DBUSERNAME = os.environ.get("DB_USERNAME")
DBPASSSWORD = os.environ.get("DB_PASSWORD")

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = 'localhost:9092',
    auto_offset_reset='latest', 
    enable_auto_commit=True,
    value_deserializer=json.loads
)

# cluster = MongoClient("mongodb+srv://:@cluster0.50fxadn.mongodb.net/?retryWrites=true&w=majority")
cluster = MongoClient("mongodb://localhost:27017")
db = cluster["thesis"]
collection = db["rawRealtimeData2"]
collection2 = db["rawDailyData"]

while True:
    for mes in consumer:
        mes = mes.value
        timestamp = datetime.now()
        json.dumps(mes)
        for value in mes:
            value.update({"TimeStamp": timestamp})
        if mes:
            collection.insert_many(mes)
            print(f"send to db {timestamp}")

# consumer2 = KafkaConsumer(
#     'raw_daily',
#     bootstrap_servers = 'localhost:9092',
#     auto_offset_reset='latest', 
#     enable_auto_commit=True,
#     value_deserializer=json.loads
# )

# while True:
#     for mes in consumer2:
#         collection2.insert_many(mes.value)
#         print(f"send daily to db")