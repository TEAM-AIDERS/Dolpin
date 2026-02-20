import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import pytest

# Allow direct execution: `python tests/test_kafka_spikeagent.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dolpin_langgraph.nodes import spike_analyzer_node
from src.schemas.kafka_schema import ContentData, KafkaMessage


def _kafka_ready() -> bool:
    required = [
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_API_KEY",
        "KAFKA_API_SECRET",
        "KAFKA_TOPIC",
    ]
    return all(bool(os.getenv(k)) for k in required)


def _to_state(msg: KafkaMessage) -> Dict[str, Any]:
    collected_at = msg.collected_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "trace_id": msg.message_id,
        "spike_event": {
            "keyword": msg.keyword,
            "current_volume": 1,
            "baseline": 1,
            "spike_rate": 0.0,
            "detected_at": collected_at,
            "messages": [
                {
                    "text": msg.content_data.text if msg.content_data else "",
                    "timestamp": collected_at,
                    "source": msg.source,
                    "author_id": msg.content_data.author_id if msg.content_data else "unknown",
                    "metrics": msg.content_data.metrics if msg.content_data else {"likes": 0, "retweets": 0, "replies": 0},
                    "detected_language": "ko",
                }
            ],
        },
        "spike_analysis": None,
        "node_insights": {},
        "error_logs": [],
    }


@pytest.mark.integration
def test_real_kafka_to_spikeagent():
    if os.getenv("RUN_REAL_KAFKA", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("Set RUN_REAL_KAFKA=1 to run real Kafka integration test.")
    if not _kafka_ready():
        pytest.skip("Kafka env vars are missing.")
    Consumer = pytest.importorskip("confluent_kafka").Consumer
    from src.pipeline.kafka_producer import KafkaProducer

    topic = os.getenv("KAFKA_TOPIC")
    keyword = f"e2e-spike-{uuid.uuid4().hex[:8]}"
    payload = KafkaMessage(
        type="post",
        source="twitter",
        keyword=keyword,
        content_data=ContentData(
            text="최고다 응원한다",
            author_id="e2e_user",
            metrics={"likes": 3, "retweets": 1, "replies": 0},
        ),
    )
    payload.validate_payload()

    consumer_conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": os.getenv("KAFKA_API_KEY"),
        "sasl.password": os.getenv("KAFKA_API_SECRET"),
        "group.id": f"dolpin-e2e-{uuid.uuid4().hex[:8]}",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
    }

    consumer = Consumer(consumer_conf)
    consumer.subscribe([topic])

    # Wait until assignment is complete. With auto.offset.reset=latest,
    # producing before assignment can skip the just-produced message.
    assign_deadline = time.time() + 10
    while time.time() < assign_deadline:
        consumer.poll(0.2)
        if consumer.assignment():
            break
    assert consumer.assignment(), "Kafka consumer partition assignment timed out."

    producer = KafkaProducer()
    producer.send(payload)
    producer.flush()

    found = None
    deadline = time.time() + 60
    while time.time() < deadline:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            continue
        data = json.loads(msg.value().decode("utf-8"))
        if data.get("message_id") == payload.message_id:
            found = KafkaMessage.model_validate(data)
            break

    consumer.close()
    assert found is not None, "Did not receive produced message from Kafka in time."

    result = spike_analyzer_node(_to_state(found))
    spike = result.get("spike_analysis")
    assert spike is not None
    assert spike["spike_rate"] == 1.0
    assert spike["spike_nature"] in {"positive", "neutral", "mixed", "negative"}
    assert result["node_insights"].get("spike_analyzer")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
