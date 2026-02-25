from src.dolpin_langgraph.nodes import router2_node, causality_node


def test_sentiment_to_causality_full_analysis():
    state = {
        "trace_id": "t-causality",
        "node_insights": {},
        "error_logs": [],
        "spike_event": {
            "keyword": "테스트",
            "messages": [
                {
                    "text": "불매해야 한다",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "source": "twitter",
                    "author_id": "u1",
                    "metrics": {"likes": 3, "retweets": 1, "replies": 0},
                    "detected_language": "ko",
                }
            ],
        },
        "spike_analysis": {
            "is_significant": True,
            "spike_nature": "negative",
            "spike_rate": 3.2,
            "actionability_score": 0.6,
        },
        "sentiment_result": {
            "dominant_sentiment": "boycott",
            "sentiment_distribution": {
                "boycott": 0.3,
                "fanwar": 0.0,
                "support": 0.1,
                "disappointment": 0.6,
            },
            "confidence": 0.7,
            "sentiment_shift": "worsening",
            "analyzed_count": 1,
            "representative_messages": {"boycott": ["불매해야 한다"]},
        },
        "route2_decision": None,
        "positive_viral_detected": None,
        "causality_result": None,
    }

    state = router2_node(state)
    assert state["route2_decision"] == "full_analysis"

    state = causality_node(state)
    assert state.get("causality_result") is not None
    assert state["node_insights"].get("causality")
