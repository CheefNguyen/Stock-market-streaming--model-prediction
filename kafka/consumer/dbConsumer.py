from kafka import KafkaConsumer
import json
import pymongo
from pymongo import MongoClient
import time

TOPIC_NAME = 'raw_data'

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = 'localhost:9092',
    auto_offset_reset = 'latest',
    value_deserializer=json.loads
)

# cluster = MongoClient("mongodb+srv://nguyen7obu:iwcwLSDyE0DF22lo@cluster0.50fxadn.mongodb.net/?retryWrites=true&w=majority")
cluster = MongoClient("mongodb://localhost:27017")
db = cluster["thesis"]
collection = db["rawData"]

for mes in consumer:
    collection.insert_many(mes.value)
    print("send to db")