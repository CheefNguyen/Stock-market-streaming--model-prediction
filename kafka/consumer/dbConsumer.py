from kafka import KafkaConsumer
import json
import pymongo
from pymongo import MongoClient
import time
from dotenv import load_dotenv
import os

TOPIC_NAME = 'raw_data'
DBUSERNAME = os.environ.get("DB_USERNAME")
DBPASSSWORD = os.environ.get("DB_PASSWORD")

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = 'localhost:9092',
    auto_offset_reset = 'latest',
    value_deserializer=json.loads
)

# cluster = MongoClient("mongodb+srv://:@cluster0.50fxadn.mongodb.net/?retryWrites=true&w=majority")
cluster = MongoClient("mongodb://localhost:27017")
db = cluster["thesis"]
collection = db["rawData"]

for mes in consumer:
    collection.insert_many(mes.value)
    print("send to db")