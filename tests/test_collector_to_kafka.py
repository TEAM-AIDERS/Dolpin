import json
import os
import sys
import threading
import time
import uuid

import pytest
from dotenv import load_dotenv

# Allow direct execution: `python tests/test_collector_to_kafka.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()

def _kafka_ready() -> bool:
    required = [
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_API_KEY",
        "KAFKA_API_SECRET",
        "KAFKA_TOPIC",
    ]
    return all(bool(os.getenv(k)) for k in required)


@pytest.mark.integration
def test_real_collector_to_kafka():
    if os.getenv("RUN_REAL_COLLECTOR", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("Set RUN_REAL_COLLECTOR=1 to run real collector->Kafka integration test.")
    if not _kafka_ready():
        pytest.skip("Kafka env vars are missing.")
    Consumer = pytest.importorskip("confluent_kafka").Consumer

    topic = os.getenv("KAFKA_TOPIC")
    keyword = os.getenv("COLLECTOR_TEST_KEYWORD", "").strip() or "NCT WISH"

    consumer_conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": os.getenv("KAFKA_API_KEY"),
        "sasl.password": os.getenv("KAFKA_API_SECRET"),
        "group.id": f"dolpin-collector-e2e-{uuid.uuid4().hex[:8]}",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
    }

    consumer = Consumer(consumer_conf)
    consumer.subscribe([topic])

    assign_deadline = time.time() + 10
    while time.time() < assign_deadline:
        consumer.poll(0.2)
        if consumer.assignment():
            break
    assert consumer.assignment(), "Kafka consumer partition assignment timed out."

    pytest.importorskip("playwright")
    from src.pipeline.collector import UnifiedCollector

    collector = UnifiedCollector()
    t = threading.Thread(target=collector.run, args=(keyword,), daemon=True)
    t.start()

    found = None
    deadline = time.time() + 120
    while time.time() < deadline:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        data = json.loads(msg.value().decode("utf-8"))
        if data.get("keyword") != keyword:
            continue
        found = data
        break

    consumer.close()
    t.join(timeout=1.0)

    assert found is not None, "Collector produced no Kafka event for the test keyword."
    assert found.get("type") in {"post", "trend"}
    assert found.get("source") in {"twitter", "instiz", "google_trends", "theqoo"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
