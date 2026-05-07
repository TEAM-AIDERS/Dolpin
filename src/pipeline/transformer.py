"""
KafkaMessage → AnalysisState 변환 모듈

Kafka에서 수신한 KafkaMessage를 LangGraph 워크플로우의
초기 AnalysisState로 변환합니다.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

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


def kafka_messages_to_state(
    msgs: List[KafkaMessage],
    baseline_for_trend: int = 50,
) -> Dict[str, Any]:
    """
    여러 KafkaMessage → 하나의 AnalysisState 변환 (키워드별 배치 처리용)

    post 메시지들을 하나의 spike_event.messages로 합쳐서
    충분한 텍스트 컨텍스트로 감정/스파이크 분석이 가능하게 한다.
    trend 메시지가 섞여 있으면 spike_rate 계산에 활용한다.
    """
    if not msgs:
        raise ValueError("msgs가 비어 있습니다.")

    if len(msgs) == 1:
        return kafka_message_to_state(msgs[0], baseline_for_trend)

    keyword = msgs[0].keyword
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    combined_messages: List[Dict[str, Any]] = []
    trend_spike_rate: float = 0.0

    for msg in msgs:
        collected_at = (
            msg.collected_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if msg.collected_at
            else now_iso
        )

        if msg.type == "post" and msg.content_data:
            combined_messages.append({
                "id": str(uuid.uuid4()),
                "source_message_id": msg.message_id,
                "text": msg.content_data.text,
                "timestamp": collected_at,
                "source": msg.source,
                "author_id": msg.content_data.author_id,
                "metrics": msg.content_data.metrics,
                "is_anonymized": False,
                "detected_language": "ko",
            })

        elif msg.type == "trend" and msg.trend_data:
            # trend 메시지: 검색어를 텍스트로 추가하고 spike_rate 계산에 활용
            for q in msg.trend_data.rising_queries:
                combined_messages.append({
                    "id": str(uuid.uuid4()),
                    "source_message_id": q.query,
                    "text": q.query,
                    "timestamp": collected_at,
                    "source": msg.source,
                    "detected_language": "ko",
                })
            rate = round(msg.trend_data.interest_score / baseline_for_trend, 2) if baseline_for_trend > 0 else 0.0
            trend_spike_rate = max(trend_spike_rate, rate)

    post_count = sum(1 for m in msgs if m.type == "post")
    current_volume = post_count if post_count > 0 else int(trend_spike_rate * baseline_for_trend)
    baseline = 1 if post_count > 0 else baseline_for_trend
    # post 메시지가 있으면 SpikeAnalyzer가 current_volume/baseline으로 재계산하도록 0.0으로 세팅
    # (trend_spike_rate를 그대로 넘기면 SpikeAnalyzer가 재계산하지 않아 not_significant 처리됨)
    spike_rate = 0.0 if post_count > 0 else trend_spike_rate

    # 대표 trace_id는 첫 번째 메시지 사용
    trace_id = msgs[0].message_id

    spike_event = {
        "keyword": keyword,
        "spike_rate": spike_rate,
        "baseline": baseline,
        "current_volume": current_volume,
        "detected_at": now_iso,
        "time_window": "1h",
        "messages": combined_messages,
        "raw_kafka_message_ids": [m.message_id for m in msgs],
    }

    return {
        "spike_event": spike_event,
        "trace_id": trace_id,
        "workflow_start_time": now_iso,
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
