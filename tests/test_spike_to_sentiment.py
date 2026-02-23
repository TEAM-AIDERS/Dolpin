from src.dolpin_langgraph.graph import compile_workflow


def test_spike_to_sentiment_integration():
    workflow = compile_workflow()

    state = {
        "trace_id": "test-trace",
        "spike_event": {
            "keyword": "테스트",
            "current_volume": 10,
            "baseline": 5,
            "spike_rate": 2.0,
            "detected_at": "2026-01-01T00:00:00Z",
            "messages": [
                {
                    "text": "보이콧 해야 하는 거 아니냐",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "source": "twitter",
                    "author_id": "user1",
                    "metrics": {"likes": 3, "retweets": 1, "replies": 0},
                    "detected_language": "ko",
                }
            ],
        },
        "spike_analysis": None,
        "sentiment_result": None,
        "node_insights": {},
        "error_logs": [],
    }

    result = workflow.invoke(state)

    assert "sentiment_result" in result
    assert result["sentiment_result"] is not None

