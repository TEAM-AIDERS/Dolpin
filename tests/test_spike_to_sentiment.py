from src.dolpin_langgraph.graph import compile_workflow


def test_spike_to_sentiment_integration():
    workflow = compile_workflow()

    state = {
        "trace_id": "test-trace",
        "workflow_start_time": "2026-01-01T00:00:00Z",
        "route1_decision": None,
        "route2_decision": None,
        "route3_decision": None,
        "skipped": False,
        "skip_reason": None,
        "node_insights": {},
        "error_logs": [],
        "spike_analysis": None,
        "sentiment_result": None,
        "spike_event": {
            "keyword": "테스트",
            "current_volume": 10,
            "baseline": 5,
            "spike_rate": 2.0,
            "detected_at": "2026-01-01T00:00:00Z",
            "messages": [
                {
                    "id": "m1",
                    "source_message_id": "sm1",
                    "text": "보이콧 해야 하는 거 아니냐",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "source": "twitter",
                    "author_id": "user1",
                    "metrics": {"likes": 3, "retweets": 1, "replies": 0},
                    "is_anonymized": False,
                    "detected_language": "ko",
                }
            ],
        },
    }

    result = workflow.invoke(state)

    assert result.get("sentiment_result") is not None
    assert result["node_insights"].get("sentiment")
