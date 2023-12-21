from kafka import KafkaConsumer
import json

TOPIC_NAME = 'raw_realtime'

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = 'localhost:9092',
    auto_offset_reset = 'latest',
    value_deserializer=json.loads
)

for mes in consumer:
    print('Received:', mes.value)