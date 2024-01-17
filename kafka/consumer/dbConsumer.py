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
    bootstrap_servers = 'kafka:9092',
    auto_offset_reset='latest', 
    enable_auto_commit=True,
    value_deserializer=json.loads
)

# cluster = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")
cluster = MongoClient("mongodb://localhost:27017")
db = cluster["thesis"]
collection = db["rawRealtimeData2"]

while True:
    for mes in consumer:
        # mes = mes.value
        # timestamp = datetime.now()
        # json.dumps(mes)
        # for value in mes:
        #     value.update({"TimeStamp": timestamp})
        # if mes:
        #     collection.insert_many(mes)
        #     print(f"send to db {timestamp}")
        print(mes)
