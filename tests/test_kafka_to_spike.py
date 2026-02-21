import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# Allow direct execution: `python tests/test_kafka_to_spike.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.schemas.kafka_schema import ContentData, KafkaMessage, RegionStat, TrendData, TrendQuery


def kafka_message_to_state(msg: KafkaMessage, baseline_for_trend: int = 50) -> Dict[str, Any]:
    """Map KafkaMessage to the minimal state needed by spike_analyzer_node."""
    collected_at = (
        msg.collected_at.isoformat()
        if msg.collected_at
        else datetime.now(timezone.utc).isoformat()
    )

    if msg.type == "post" and msg.content_data:
        messages: List[Dict[str, Any]] = [
            {
                "text": msg.content_data.text,
                "timestamp": collected_at,
                "source": msg.source,
                "author_id": msg.content_data.author_id,
                "metrics": msg.content_data.metrics,
                "detected_language": "ko",
            }
        ]
        current_volume = 1
        baseline = 1
        spike_rate = 0.0  # Let agent derive from current_volume / baseline
    elif msg.type == "trend" and msg.trend_data:
        messages = [
            {
                "text": q.query,
                "timestamp": collected_at,
                "source": msg.source,
                "detected_language": "ko",
            }
            for q in msg.trend_data.rising_queries
        ]
        current_volume = msg.trend_data.interest_score
        baseline = baseline_for_trend
        spike_rate = current_volume / baseline if baseline > 0 else 0.0
    else:
        messages = []
        current_volume = 0
        baseline = 0
        spike_rate = 0.0

    return {
        "trace_id": msg.message_id,
        "spike_event": {
            "keyword": msg.keyword,
            "current_volume": current_volume,
            "baseline": baseline,
            "spike_rate": round(spike_rate, 2),
            "detected_at": collected_at,
            "messages": messages,
        },
        "spike_analysis": None,
        "node_insights": {},
        "error_logs": [],
    }


@pytest.fixture
def run_spike(monkeypatch):
    """Run spike_analyzer_node with MCP mocked and singleton reset per test."""
    mock_mcp = MagicMock()
    mock_mcp.lexicon_analyze.return_value = {"matched_terms": []}

    import src.dolpin_langgraph.nodes as nodes
    import src.agents.spike_analyzer as spike_module

    monkeypatch.setattr(spike_module, "get_mcp_client", lambda: mock_mcp)
    nodes._SPIKE_ANALYZER_AGENT = None

    def _run(msg: KafkaMessage) -> Dict[str, Any]:
        state = kafka_message_to_state(msg)
        return nodes.spike_analyzer_node(state)

    return _run


def test_post_type_flows_to_spike_agent(run_spike):
    msg = KafkaMessage(
        type="post",
        source="twitter",
        keyword="BTS",
        content_data=ContentData(
            text="최고다 응원한다",
            author_id="user_001",
            metrics={"likes": 500, "retweets": 200, "replies": 50},
        ),
    )
    msg.validate_payload()

    result = run_spike(msg)
    sa = result["spike_analysis"]

    assert sa is not None
    assert sa["spike_rate"] == 1.0
    assert sa["spike_nature"] == "positive"
    assert sa["is_significant"] is False
    assert sa["data_completeness"] == "confirmed"


def test_trend_type_flows_to_spike_agent(run_spike):
    msg = KafkaMessage(
        type="trend",
        source="google_trends",
        keyword="BTS",
        trend_data=TrendData(
            interest_score=90,
            is_partial=False,
            rising_queries=[
                TrendQuery(query="BTS 컴백", value="+500%"),
                TrendQuery(query="BTS 신곡", value="+300%"),
            ],
            top_queries=["BTS", "방탄소년단"],
            region_stats=[RegionStat(geo="KR", value=100)],
        ),
    )
    msg.validate_payload()

    result = run_spike(msg)
    sa = result["spike_analysis"]

    assert sa is not None
    assert sa["spike_rate"] == 1.8
    assert sa["is_significant"] is False
    assert "google_trends" in sa["viral_indicators"]["cross_platform"]


def test_negative_post_sets_negative_nature(run_spike):
    msg = KafkaMessage(
        type="post",
        source="twitter",
        keyword="BTS",
        content_data=ContentData(
            text="불매 실망 최악",
            author_id="user_002",
            metrics={"likes": 10, "retweets": 5, "replies": 2},
        ),
    )
    msg.validate_payload()

    result = run_spike(msg)
    sa = result["spike_analysis"]

    assert sa is not None
    assert sa["spike_nature"] == "negative"


def test_invalid_payload_raises():
    msg = KafkaMessage(type="post", source="twitter", keyword="BTS")
    with pytest.raises(ValueError):
        msg.validate_payload()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
