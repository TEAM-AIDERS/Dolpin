import os
import sys
from unittest.mock import MagicMock

import pytest

# Allow direct execution: `python tests/test_kafka_to_spike.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.transformer import kafka_message_to_state
from src.schemas.kafka_schema import ContentData, KafkaMessage, RegionStat, TrendData, TrendQuery


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
