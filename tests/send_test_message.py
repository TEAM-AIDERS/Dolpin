import json
import os
import time
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

producer_conf = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "PLAIN",
    "sasl.username": os.getenv("KAFKA_API_KEY"),
    "sasl.password": os.getenv("KAFKA_API_SECRET"),
}

producer = Producer(producer_conf)

topic = os.getenv("KAFKA_TOPIC")

message = {
    "keyword": "위시",
    "type": "post",
    "source": "test",
    "content_data": {
        "id": f"test-{int(time.time() * 1000)}",
        "text": "위시 관련 테스트 메시지",
        "author_id": "tester",
        "metrics": {
            "likes": 1,
            "retweets": 0
        }
    },
    "collected_at": "2026-03-19T00:00:00Z"
}

producer.produce(topic, value=json.dumps(message).encode("utf-8"))
producer.flush()

print(f"sent to {topic}")
