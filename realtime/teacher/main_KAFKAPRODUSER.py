from kafka import KafkaProducer
import psutil
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'system-metrics'

while True:
    metrics = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory': psutil.virtual_memory()._asdict(),
        'disk': psutil.disk_usage('/')._asdict()
    }
    producer.send(topic, metrics)
    print(f"Sent: {metrics}")
    time.sleep(2)  # Adjust interval as needed
