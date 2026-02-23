from src.dolpin_langgraph.graph import compile_workflow


def test_spike_to_sentiment():
    workflow = compile_workflow()

    state = {
        "issue_id": "sentiment-test",
        "spike_event": {
            "messages": [
                {"id": "1", "text": "보이콧 해야 하는 거 아니냐"},
                {"id": "2", "text": "회사 대응 진짜 실망이다"},
            ]
        },
    }

    result = workflow.invoke(state)

    assert "sentiment_result" in result
    assert result["sentiment_result"] is not None
