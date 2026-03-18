"""
KafkaMessage → AnalysisState 변환 모듈

Kafka에서 수신한 KafkaMessage를 LangGraph 워크플로우의
초기 AnalysisState로 변환합니다.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from src.schemas.kafka_schema import KafkaMessage


def kafka_message_to_state(
    msg: KafkaMessage,
    baseline_for_trend: int = 50,
) -> Dict[str, Any]:
    """
    KafkaMessage → AnalysisState 초기 상태 변환

    Args:
        msg: Kafka에서 수신한 메시지 (type="post" | "trend")
        baseline_for_trend: trend 타입일 때 사용할 baseline (기본 50)

    Returns:
        LangGraph spike_analyzer_node에 바로 전달 가능한 AnalysisState dict
    """
    collected_at = (
        msg.collected_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if msg.collected_at
        else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    if msg.type == "post" and msg.content_data:
        messages = [
            {
                "id": str(uuid.uuid4()),
                "source_message_id": msg.message_id,
                "text": msg.content_data.text,
                "timestamp": collected_at,
                "source": msg.source,
                "author_id": msg.content_data.author_id,
                "metrics": msg.content_data.metrics,
                "is_anonymized": False,
                "detected_language": "ko",
            }
        ]
        current_volume = 1
        baseline = 1
        spike_rate = 0.0  # SpikeAnalyzerAgent가 baseline_history로 재계산

    elif msg.type == "trend" and msg.trend_data:
        messages = [
            {
                "id": str(uuid.uuid4()),
                "source_message_id": q.query,
                "text": q.query,
                "timestamp": collected_at,
                "source": msg.source,
                "detected_language": "ko",
            }
            for q in msg.trend_data.rising_queries
        ]
        current_volume = msg.trend_data.interest_score
        baseline = baseline_for_trend
        spike_rate = round(current_volume / baseline, 2) if baseline > 0 else 0.0

    else:
        messages = []
        current_volume = 0
        baseline = 0
        spike_rate = 0.0

    spike_event = {
        "keyword": msg.keyword,
        "spike_rate": spike_rate,
        "baseline": baseline,
        "current_volume": current_volume,
        "detected_at": collected_at,
        "time_window": "1h",
        "messages": messages,
        "raw_kafka_message_ids": [msg.message_id],
    }

    return {
        "spike_event": spike_event,
        "trace_id": msg.message_id,
        "workflow_start_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sentiment_model_path": None,
        "device": "cpu",
        "lexicon_lookup_raw": None,
        "route1_decision": None,
        "route2_decision": None,
        "route3_decision": None,
        "positive_viral_detected": None,
        "spike_analysis": None,
        "lexicon_matches": None,
        "sentiment_result": None,
        "causality_result": None,
        "legal_risk": None,
        "amplification_summary": None,
        "playbook": None,
        "node_insights": {},
        "executive_brief": None,
        "error_logs": [],
        "skipped": False,
        "skip_reason": None,
    }
