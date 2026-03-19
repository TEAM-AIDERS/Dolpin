import json
import os
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
    "keyword": "워시",
    "type": "post",
    "source": "test",
    "content_data": {
        "id": "test-1",
        "text": "워시 관련 테스트 메시지",
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
