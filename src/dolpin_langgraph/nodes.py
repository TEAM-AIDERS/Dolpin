"""
LangGraph 노드 래퍼 함수
각 에이전트를 호출하고 State를 업데이트합니다.

버전: v2 (260203)
- state 저장 형태 확장 반영
- node_insights 키를 graph node id와 동일하게 통일
- lexicon_lookup는 MCPClient singleton(get_mcp_client) 사용
"""

import logging, os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .state import AnalysisState, LegalRAGInput, LegalRiskResult, ErrorLog
from .edges import (
    route_after_spike_analysis,
    route_after_sentiment,
    route_after_causality
)

from src.agents.sentiment_agent import build_agent as build_sentiment_agent
from src.agents.legalrag_agent import check_legal_risk

logger = logging.getLogger(__name__)


# ============================================================
# 공통 유틸
# ============================================================

def _ensure_state_collections(state: AnalysisState) -> None:
    """state에 기본 컬렉션들이 없으면 초기화"""
    if "error_logs" not in state or state["error_logs"] is None:
        state["error_logs"] = []
    if "node_insights" not in state or state["node_insights"] is None:
        state["node_insights"] = {}


def _add_error_log(
    state: AnalysisState,
    stage: str,
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """에러 로그 추가"""
    _ensure_state_collections(state)

    error_log: ErrorLog = {
        "stage": stage,  # state.py Literal에 맞춰야 함
        "error_type": error_type,
        "message": message,
        "occurred_at": datetime.utcnow().isoformat() + "Z",
        "trace_id": state.get("trace_id", "unknown"),
        "details": details
    }
    state["error_logs"].append(error_log)
    logger.error(f"[{stage}] {error_type}: {message}", extra={"trace_id": state.get("trace_id", "unknown")})


def _update_node_insight(state: AnalysisState, node_id: str, insight: str) -> None:
    """node_insights 업데이트 (node_id는 graph.add_node 이름과 동일하게)"""
    _ensure_state_collections(state)
    state["node_insights"][node_id] = insight
    logger.info(f"[{node_id}] {insight}", extra={"trace_id": state.get("trace_id", "unknown")})


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_existing_path(candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        raw = str(candidate or "").strip()
        if raw and Path(raw).exists():
            return raw
    return None


def _should_hard_fail(state: AnalysisState, stage: str) -> bool:
    raw = state.get("hard_fail_stages")
    if isinstance(raw, (list, tuple, set)):
        stages = {str(v).strip() for v in raw if str(v).strip()}
    else:
        env_raw = os.getenv("DOLPIN_HARD_FAIL_STAGES", "")
        stages = {s.strip() for s in env_raw.split(",") if s.strip()}
    hard_fail_all = str(os.getenv("DOLPIN_HARD_FAIL_ALL", "")).lower() in {"1", "true", "yes", "on"}
    return hard_fail_all or stage in stages


def _raise_if_hard_fail(state: AnalysisState, stage: str, message: str, exc: Optional[Exception] = None) -> None:
    if not _should_hard_fail(state, stage):
        return
    if exc is None:
        raise RuntimeError(f"[hard-fail:{stage}] {message}")
    raise RuntimeError(f"[hard-fail:{stage}] {message}") from exc



# ============================================================
# spike_analyzer
# ============================================================

def spike_analyzer_node(state: AnalysisState) -> AnalysisState:
    """
    급등 분석 노드
    - spike_event 기반으로 significance/viral/actionability 계산
    - 임계값 override 지원:
      - significant_spike_threshold (default: 2.0)
      - breakout_spike_threshold (default: 3.0)
    """
    _ensure_state_collections(state)

    try:
        spike_event = state.get("spike_event") or {}
        messages = spike_event.get("messages", []) or []
        keyword = str(spike_event.get("keyword", "unknown"))

        baseline = int(spike_event.get("baseline", 0) or 0)
        current_volume = int(spike_event.get("current_volume", 0) or 0)
        spike_rate_raw = float(spike_event.get("spike_rate", 0.0) or 0.0)
        if spike_rate_raw <= 0 and baseline > 0 and current_volume > 0:
            spike_rate_raw = current_volume / baseline
        spike_rate = round(float(spike_rate_raw), 2)

        sig_threshold = float(state.get("significant_spike_threshold", 2.0) or 2.0)
        breakout_threshold = float(state.get("breakout_spike_threshold", 3.0) or 3.0)
        is_significant = spike_rate >= sig_threshold
        has_breakout = spike_rate >= breakout_threshold

        source_set = {str((m or {}).get("source", "")).strip() for m in messages if (m or {}).get("source")}
        cross_platform = sorted(source_set)

        engagement_sum = 0
        for msg in messages:
            metrics = (msg or {}).get("metrics", {}) or {}
            engagement_sum += int(metrics.get("likes", 0) or 0)
            engagement_sum += int(metrics.get("retweets", 0) or 0)
            engagement_sum += int(metrics.get("replies", 0) or 0)

        non_ko_count = 0
        for msg in messages:
            lang = str((msg or {}).get("detected_language", "")).lower().strip()
            if lang and lang not in ("ko", "kr"):
                non_ko_count += 1
        international_reach = round(non_ko_count / max(1, len(messages)), 2)

        positive_cues = ("좋", "최고", "감사", "응원", "축하", "사랑", "행복", "대박")
        negative_cues = ("불매", "보이콧", "실망", "화나", "논란", "문제", "싫", "최악", "탈덕")
        positive_hits = 0
        negative_hits = 0
        for msg in messages:
            text = str((msg or {}).get("text", "")).lower()
            if any(cue in text for cue in positive_cues):
                positive_hits += 1
            if any(cue in text for cue in negative_cues):
                negative_hits += 1

        if positive_hits > 0 and negative_hits > 0:
            spike_nature = "mixed"
        elif negative_hits > positive_hits:
            spike_nature = "negative"
        elif positive_hits > 0:
            spike_nature = "positive"
        else:
            spike_nature = "neutral"

        timestamp_candidates = []
        for msg in messages:
            ts = (msg or {}).get("timestamp")
            if ts:
                timestamp_candidates.append(str(ts))
        detected_at = spike_event.get("detected_at")
        if detected_at:
            timestamp_candidates.append(str(detected_at))
        peak_timestamp = max(timestamp_candidates) if timestamp_candidates else _utcnow_z()

        duration_minutes = 0
        if len(messages) >= 2:
            parsed = []
            for msg in messages:
                ts = (msg or {}).get("timestamp")
                if not ts:
                    continue
                try:
                    parsed.append(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
                except Exception:
                    continue
            if len(parsed) >= 2:
                duration_minutes = int((max(parsed) - min(parsed)).total_seconds() / 60)

        actionability_score = (
            0.45 * min(1.0, spike_rate / 5.0)
            + 0.25 * min(1.0, len(messages) / 50.0)
            + 0.20 * (1.0 if has_breakout else 0.0)
            + 0.10 * min(1.0, len(cross_platform) / 3.0)
        )
        actionability_score = round(max(0.0, min(1.0, actionability_score)), 2)

        confidence = (
            0.50
            + 0.20 * min(1.0, len(messages) / 30.0)
            + 0.20 * (1.0 if baseline > 0 and current_volume > 0 else 0.0)
            + 0.10 * min(1.0, len(cross_platform) / 2.0)
        )
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        data_completeness = "confirmed" if (baseline > 0 and current_volume > 0 and messages) else "partial"
        partial_data_warning = None
        if data_completeness != "confirmed":
            partial_data_warning = "incomplete spike_event fields"

        max_rise_rate = "Breakout" if has_breakout else f"+{max(0.0, (spike_rate - 1.0) * 100):.0f}%"
        breakout_queries = [keyword] if has_breakout and keyword != "unknown" else []
        is_trending = has_breakout or len(cross_platform) >= 2

        result = {
            "is_significant": is_significant,
            "spike_rate": spike_rate,
            "spike_type": "organic",
            "spike_nature": spike_nature,
            "peak_timestamp": peak_timestamp,
            "duration_minutes": duration_minutes,
            "confidence": confidence,
            "actionability_score": actionability_score,
            "data_completeness": data_completeness,
            "partial_data_warning": partial_data_warning,
            "viral_indicators": {
                "is_trending": is_trending,
                "has_breakout": has_breakout,
                "max_rise_rate": max_rise_rate,
                "breakout_queries": breakout_queries,
                "cross_platform": cross_platform,
                "international_reach": international_reach
            }
        }

        state["spike_analysis"] = result

        insight = (
            f"{result['spike_rate']}x spike, nature={result['spike_nature']}, "
            f"actionability={result['actionability_score']}, engagement={engagement_sum}"
        )
        if result.get("partial_data_warning"):
            insight += f", {result['partial_data_warning']}"
        _update_node_insight(state, "spike_analyzer", insight)

        logger.info(f"SpikeAnalyzer 완료: is_significant={result['is_significant']}")
        return state

    except Exception as e:
        _add_error_log(
            state,
            stage="spike_analyzer",
            error_type="exception",
            message=str(e),
            details={"keyword": state.get("spike_event", {}).get("keyword")}
        )

        # 실패 시에도 기본값 설정
        state["spike_analysis"] = {
            "is_significant": False,
            "spike_rate": 0.0,
            "spike_type": "organic",
            "spike_nature": "neutral",
            "peak_timestamp": _utcnow_z(),
            "duration_minutes": 0,
            "confidence": 0.0,
            "actionability_score": 0.0,
            "data_completeness": "partial",
            "partial_data_warning": "분석 실패",
            "viral_indicators": {
                "is_trending": False,
                "has_breakout": False,
                "max_rise_rate": "0%",
                "breakout_queries": [],
                "cross_platform": [],
                "international_reach": 0.0
            }
        }
        _update_node_insight(state, "spike_analyzer", "급등 분석 중 오류가 발생했습니다.")
        return state
# router1
# ============================================================

def router1_node(state: AnalysisState) -> AnalysisState:
    """Router 1차: Skip vs Analyze"""
    _ensure_state_collections(state)

    try:
        decision = route_after_spike_analysis(state)
        state["route1_decision"] = decision

        if decision == "skip":
            state["skipped"] = True
            state["skip_reason"] = "not_significant"
            _update_node_insight(state, "router1", "skip: not_significant")
        else:
            state["skipped"] = False
            state["skip_reason"] = None
            _update_node_insight(state, "router1", "analyze: significant")

        return state

    except Exception as e:
        _add_error_log(state, "spike_analyzer", "exception", f"Router1 에러: {str(e)}")
        state["route1_decision"] = "skip"
        state["skipped"] = True
        state["skip_reason"] = "not_significant"
        _update_node_insight(state, "router1", "skip: exception")
        return state


# ============================================================
# lexicon_lookup
# ============================================================

def lexicon_lookup_node(state: AnalysisState) -> AnalysisState:
    """
    렉시콘 매칭 노드
    - MCPClient singleton(get_mcp_client) 사용
    - state["lexicon_matches"]: Optional[Dict[str, LexiconMatch]]
      LexiconMatch = { "count": int, "type": str, "terms": List[str] }
    """
    _ensure_state_collections(state)

    try:
        spike_event = state.get("spike_event")
        messages = (spike_event or {}).get("messages", [])
        if not messages:
            state["lexicon_matches"] = None
            _update_node_insight(state, "lexicon_lookup", "분석할 메시지 없음")
            return state

        # singleton 사용
        from src.server.mcp_client import get_mcp_client
        lexicon_path = state.get("lexicon_path", "custom_lexicon.csv")
        mcp = get_mcp_client(lexicon_path)

        # type -> LexiconMatch
        lexicon_matches_result: Dict[str, Dict[str, Any]] = {}

        all_matches = []
        
        for msg in messages:
            text = (msg or {}).get("text", "")
            if not text:
                continue

            res = mcp.lexicon_analyze(text) or {}
            matches = res.get("matches", [])  # dict 리턴 기준
            
            all_matches.extend(matches)

            for match in matches:
                type_name = match.get("type", "unknown")
                term = match.get("term")

                if type_name not in lexicon_matches_result:
                    lexicon_matches_result[type_name] = {
                        "count": 0,
                        "type": type_name,
                        "terms": []
                    }

                lexicon_matches_result[type_name]["count"] += 1
                if term and term not in lexicon_matches_result[type_name]["terms"]:
                    lexicon_matches_result[type_name]["terms"].append(term)

        state["lexicon_matches"] = lexicon_matches_result if lexicon_matches_result else None
        state["lexicon_lookup_raw"] = {"matches": all_matches} if all_matches else None


        if lexicon_matches_result:
            top = sorted(
                lexicon_matches_result.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:3]
            summary = ", ".join([f"{k}({v['count']})" for k, v in top])
            _update_node_insight(state, "lexicon_lookup", f"렉시콘 매칭: {summary}")
        else:
            _update_node_insight(state, "lexicon_lookup", "렉시콘 매칭 결과 없음")

        logger.info(f"LexiconLookup 완료: {len(lexicon_matches_result)} 타입")
        return state

    except Exception as e:
        _add_error_log(
            state,
            stage="lexicon_lookup",  
            error_type="exception",
            message=f"LexiconLookup 실패: {str(e)}",
            details={"keyword": state.get("spike_event", {}).get("keyword")}
        )
        state["lexicon_matches"] = None
        _update_node_insight(state, "lexicon_lookup", "렉시콘 매칭 실패(예외)")
        _raise_if_hard_fail(state, "lexicon_lookup", "lexicon lookup failed", e)
        return state


# ============================================================
# sentiment
# ============================================================

def sentiment_node(state: AnalysisState) -> AnalysisState:
    """감정 분석 노드"""
    _ensure_state_collections(state)

    try:
        spike_event = state.get("spike_event")
        messages = (spike_event or {}).get("messages", [])
        if not messages:
            state["sentiment_result"] = None
            _update_node_insight(state, "sentiment", "분석할 메시지 없음")
            return state

        combined_text = " ".join([(m or {}).get("text", "") for m in messages]).strip()
        if not combined_text:
            state["sentiment_result"] = None
            _update_node_insight(state, "sentiment", "텍스트가 비어 있음")
            return state

        model_path = (
            state.get("sentiment_model_path")
            or os.getenv("SENTIMENT_MODEL_PATH")
            or "Aerisbin/sentiment-agent-v1"
        )

        device = state.get("device") or "cpu"

        try:
            agent = build_sentiment_agent(model_path, device)
        except Exception as e:
            logger.error(
                "SentimentAgent init failed: %s | model_path=%s (exists=%s) | device=%s",
                e,
                model_path,
                device,
            )
            _add_error_log(
                state,
                stage="sentiment",
                error_type="exception",
                message=f"sentiment init failed: {e}",
                details={
                    "model_path": model_path,
                    "device": device,
                },
            )
            state["sentiment_result"] = None
            _update_node_insight(state, "sentiment", "모델 초기화 실패")
            _raise_if_hard_fail(state, "sentiment", "sentiment model init failed", e)
            return state

        # 감정 분석 실행
        analyzed_count = len(messages)
        lexicon_context = state.get("lexicon_lookup_raw")
        result, route_meta = agent.analyze(
            combined_text,
            analyzed_count=analyzed_count,
            lexicon_context=lexicon_context,
        )

        # state에 필요한 필드 추가
        result["analyzed_count"] = analyzed_count
        result["confidence"] = result.get("confidence", 0.5)
        result["sentiment_shift"] = result.get("sentiment_shift", "stable")

        # representative_messages가 이미 포맷팅되어 있을 수도 있으니 안전하게
        if "representative_messages" not in result or not result["representative_messages"]:
            dom = result.get("dominant_sentiment", "support")
            result["representative_messages"] = {dom: [combined_text[:200]]}

        state["sentiment_result"] = result

        # Insight 생성
        dominant = result.get("dominant_sentiment", "unknown")
        confidence = float(result.get("confidence", 0.0))
        _update_node_insight(
            state,
            "sentiment",
            f"{dominant} {confidence*100:.0f}% (신뢰도), {analyzed_count}건"
        )

        logger.info(
            f"Sentiment 완료: dominant={dominant}, confidence={confidence:.2f}, "
            f"route={route_meta.get('route', 'unknown')}"
        )
        return state

    except Exception as e:
        _add_error_log(
            state,
            stage="sentiment",
            error_type="exception",
            message=str(e),
            details={"keyword": state.get("spike_event", {}).get("keyword")}
        )
        state["sentiment_result"] = None
        _update_node_insight(state, "sentiment", "감정 분석 실패(예외)")
        _raise_if_hard_fail(state, "sentiment", "sentiment analysis failed", e)
        return state


# ============================================================
# router2
# ============================================================

def router2_node(state: AnalysisState) -> AnalysisState:
    """
    Router 2차: Sentiment Only vs Full Analysis
    
    Note: positive_viral_detected도 여기서 설정 (Router 3차에서 재사용)
    """
    _ensure_state_collections(state)

    try:
        spike_analysis = state.get("spike_analysis")
        sentiment_result = state.get("sentiment_result")

        if not spike_analysis or not sentiment_result:
            state["route2_decision"] = "sentiment_only"
            state["positive_viral_detected"] = False
            _update_node_insight(state, "router2", "sentiment_only (missing inputs)")
            return state

        # positive_viral_detected 먼저 계산
        from .edges import _check_opportunity_signals
        has_opportunity = _check_opportunity_signals(spike_analysis, sentiment_result)
        state["positive_viral_detected"] = has_opportunity

        decision = route_after_sentiment(state)
        state["route2_decision"] = decision

        _update_node_insight(
            state,
            "router2",
            f"{decision}, positive_viral={has_opportunity}, actionability={spike_analysis.get('actionability_score')}"
        )
        return state

    except Exception as e:
        _add_error_log(state, "sentiment", "exception", f"Router2 에러: {str(e)}")
        state["route2_decision"] = "sentiment_only"
        state["positive_viral_detected"] = False
        _update_node_insight(state, "router2", "sentiment_only (exception)")
        return state


# ============================================================
# causality
# ============================================================

def causality_node(state: AnalysisState) -> AnalysisState:
    """
    인과관계 분석 노드
    - spike_event.messages를 CausalityAgent 입력으로 변환
    - 결과를 CausalityAnalysisResult 스키마로 정규화
    """
    _ensure_state_collections(state)

    try:
        spike_event = state.get("spike_event") or {}
        messages = (spike_event.get("messages") or [])
        if not messages:
            state["causality_result"] = None
            _update_node_insight(state, "causality", "분석할 메시지 없음")
            return state

        sentiment = state.get("sentiment_result") or {}
        dominant = str(sentiment.get("dominant_sentiment", "")).strip()
        dominant_type_map = {
            "support": "support_action",
            "meme": "meme_positive",
            "boycott": "boycott_action",
            "fanwar": "fanwar_action",
            "disappointment": "meme_negative",
        }
        dominant_type = dominant_type_map.get(dominant, "")

        negative_cues = ("불매", "보이콧", "실망", "논란", "탈덕", "최악")
        fanwar_cues = ("팬덤", "팬들", "싸움", "전쟁", "저격")
        meme_cues = ("밈", "짤", "드립", "웃김")
        support_cues = ("응원", "감사", "최고", "좋다")

        enriched_messages = []
        for msg in messages:
            message = dict(msg or {})
            msg_types = list(message.get("types") or [])
            text = str(message.get("text", ""))

            if not msg_types:
                if any(c in text for c in negative_cues):
                    msg_types.append("meme_negative")
                if any(c in text for c in fanwar_cues):
                    msg_types.append("fanwar_target")
                if any(c in text for c in meme_cues):
                    msg_types.append("meme_positive")
                if any(c in text for c in support_cues):
                    msg_types.append("support_action")
                if not msg_types and dominant_type:
                    msg_types = [dominant_type]

            message["types"] = msg_types
            enriched_messages.append(message)

        causality_input = dict(spike_event)
        causality_input["messages"] = enriched_messages

        from src.agents.causality_agent import analyze_network
        raw = analyze_network(causality_input)

        graph_analysis = ((raw.get("debug") or {}).get("graph_analysis") or {})
        central_nodes = graph_analysis.get("central_nodes") or []
        hub_accounts = []
        for node in central_nodes[:5]:
            degree = float(node.get("degree", 0.0) or 0.0)
            betweenness = float(node.get("betweenness", 0.0) or 0.0)
            hub_accounts.append(
                {
                    "account_id": str(node.get("id", "unknown")),
                    "influence_score": round(max(degree, betweenness), 2),
                    "follower_count": 0,
                    "account_type": "general",
                }
            )

        ts_candidates = []
        for msg in messages:
            ts = (msg or {}).get("timestamp")
            if ts:
                ts_candidates.append(str(ts))
        estimated_origin_time = min(ts_candidates) if ts_candidates else None

        result = {
            "trigger_source": raw.get("trigger_source", "organic"),
            "hub_accounts": hub_accounts,
            "retweet_network_metrics": {
                "centralization": float((raw.get("retweet_network_metrics") or {}).get("centralization", 0.0) or 0.0),
                "avg_degree": float((raw.get("retweet_network_metrics") or {}).get("avg_degree", 0.0) or 0.0),
            },
            "cascade_pattern": raw.get("cascade_pattern", "echo_chamber"),
            "estimated_origin_time": estimated_origin_time,
            "key_propagation_paths": list(raw.get("key_propagation_paths") or []),
        }

        state["causality_result"] = result
        _update_node_insight(
            state,
            "causality",
            f"{result['trigger_source']} 주도, {result['cascade_pattern']} 패턴, hubs={len(hub_accounts)}",
        )
        return state

    except Exception as e:
        _add_error_log(state, "causality", "exception", str(e))
        state["causality_result"] = None
        _update_node_insight(state, "causality", "인과 분석 실패(예외)")
        _raise_if_hard_fail(state, "causality", "causality analysis failed", e)
        return state


# ============================================================
# router3
# ============================================================

def router3_node(state: AnalysisState) -> AnalysisState:
    """Router 3차: Legal vs Amplification"""
    _ensure_state_collections(state)

    try:
        decision = route_after_causality(state)
        state["route3_decision"] = decision
        _update_node_insight(state, "router3", decision)
        return state

    except Exception as e:
        _add_error_log(state, "causality", "exception", f"Router3 에러: {str(e)}")
        state["route3_decision"] = "legal"
        _update_node_insight(state, "router3", "legal (exception)")
        return state


# ============================================================
# legal_rag
# ============================================================


async def legal_rag_node(state: AnalysisState) -> AnalysisState:
    try:
        # 1️. LegalRAGInput 구성
        spike_event = state["spike_event"]
        spike_analysis = state.get("spike_analysis")
        sentiment_result = state.get("sentiment_result")
        
        messages_text = [m["text"] for m in spike_event["messages"]]

        legal_input: LegalRAGInput = {
            "messages": messages_text,
            "spike_nature": spike_analysis["spike_nature"] if spike_analysis else "neutral",
            "dominant_sentiment": sentiment_result["dominant_sentiment"] if sentiment_result else "neutral",
            "keyword": spike_event["keyword"],
            "spike_rate": spike_analysis["spike_rate"] if spike_analysis else 0.0,
            "fanwar_targets": sentiment_result.get("fanwar_targets") if sentiment_result else None,
        }

        # 2️. LegalRAGAgent 실행
        legal_result: LegalRiskResult = await check_legal_risk(legal_input)

        # 3️. node_insights 요약
        insight_summary = (
            f"Legal Risk Level: {legal_result['overall_risk_level']} | "
            f"Clearance: {legal_result['clearance_status']}"
        )

        return {
            **state,
            "legal_risk": legal_result,
            "node_insights": {
                **state.get("node_insights", {}),
                "LegalRAG": insight_summary
            }
        }

    except Exception as e:
        # Fallback LegalRiskResult 
        fallback_result: LegalRiskResult = {
            "overall_risk_level": "medium",
            "clearance_status": "review_needed",
            "confidence": 0.2,  # 매우 낮은 신뢰도
            "rag_required": True,
            "rag_performed": False,
            "rag_confidence": None,
            "risk_assessment": None,
            "recommended_action": [
                "법률 분석 실패 - 내부 법무팀 수동 검토 필요"
            ],
            "referenced_documents": [],
            "signals": None
        }

        error_log: ErrorLog = {
            "stage": "legal_rag",
            "error_type": "exception",
            "message": str(e),
            "occurred_at": datetime.utcnow().isoformat() + "Z",
            "trace_id": state["trace_id"],
            "details": None,
        }

        return {
            **state,
            "legal_risk": fallback_result, 
            "error_logs": state.get("error_logs", []) + [error_log],
            "node_insights": {
                **state.get("node_insights", {}),
                "LegalRAG": "Legal analysis failed - fallback applied"
            }
        }


# ============================================================
# amplification
# ============================================================

def amplification_node(state: AnalysisState) -> AnalysisState:
    """긍정 바이럴 기회 요약 노드"""
    _ensure_state_collections(state)
    
    try:
        causality = state.get("causality_result") or {}
        sentiment = state.get("sentiment_result") or {}
        spike = state.get("spike_analysis") or {}
        
        viral_indicators = spike.get("viral_indicators", {})
        
        # ===== Hub Accounts =====
        hub_accounts = []
        for hub in (causality.get("hub_accounts") or [])[:5]:
            hub_accounts.append({
                "account_id": hub["account_id"],
                "influence_score": hub["influence_score"],
                "account_type": hub.get("account_type", "general"),
                "follower_count": hub.get("follower_count", 0)
            })
        
        # ===== Representative Messages =====
        support_msgs = (sentiment.get("representative_messages", {}).get("support") or [])
        representative_messages = [
            {"text": msg}  # text만!
            for msg in support_msgs[:5]
        ]
        
        # ===== 결과 =====
        result = {
            "top_platforms": viral_indicators.get("cross_platform", ["twitter"]),
            "hub_accounts": hub_accounts,
            "representative_messages": representative_messages
        }
        
        state["amplification_summary"] = result
        
        platform_count = len(result["top_platforms"])
        hub_count = len(result["hub_accounts"])
        msg_count = len(result["representative_messages"])
        _update_node_insight(
            state, 
            "amplification", 
            f"{platform_count}개 플랫폼, {hub_count}개 허브, {msg_count}개 메시지"
        )
        
        return state
    
    except Exception as e:
        _add_error_log(state, "amplification", "exception", str(e))
        state["amplification_summary"] = None
        _update_node_insight(state, "amplification", "실패")
        return state

# ============================================================
# playbook
# ============================================================

def playbook_node(state: AnalysisState) -> AnalysisState:
    """대응 전략 생성 노드"""
    _ensure_state_collections(state)

    try:
        from src.agents.playbook_agent import generate_strategy

        route2 = state.get("route2_decision")
        api_key_exists = bool(os.getenv("OPENAI_API_KEY"))

        # full_analysis 경로 → LLM 사용
        use_llm = (route2 == "full_analysis" and api_key_exists)

        playbook_result = generate_strategy(state, use_llm_enhancement=use_llm)
        state["playbook"] = playbook_result

        situation_type = playbook_result.get("situation_type", "unknown")
        priority = playbook_result.get("priority", "unknown")
        action_count = len(playbook_result.get("recommended_actions", []))

        _update_node_insight(state, "playbook", f"{situation_type}/{priority}, actions={action_count}")
        return state

    except Exception as e:
        _add_error_log(state, "playbook", "exception", str(e))
        state["playbook"] = None
        _update_node_insight(state, "playbook", "생성 실패(예외)")
        return state


# ============================================================
# exec_brief
# ============================================================

def exec_brief_node(state: AnalysisState) -> AnalysisState:
    """ExecBrief 노드 - Slack 전송을 위한 리포트 생성"""
    _ensure_state_collections(state)

    try:
        workflow_start_raw = state.get("workflow_start_time")
        duration = 0.0
        if workflow_start_raw:
            workflow_start = datetime.fromisoformat(workflow_start_raw.replace("Z", "+00:00"))
            duration = (datetime.now(workflow_start.tzinfo) - workflow_start).total_seconds()

        spike_summary = _generate_spike_summary(state)
        sentiment_summary = _generate_sentiment_summary(state)
        legal_summary = _generate_legal_summary(state)
        action_summary = _generate_action_summary(state)
        opportunity_summary = _generate_opportunity_summary(state)

        playbook = state.get("playbook") or {}
        playbook_status = "failed"
        if playbook:
            playbook_status = "success" if playbook.get("recommended_actions") else "partial"

        analysis_status = {
            "spike_analyzer": "success" if state.get("spike_analysis") else "failed",
            "sentiment": "success" if state.get("sentiment_result") else "failed",
            "causality": "success" if state.get("causality_result") else "skipped",
            "legal_rag": "success" if state.get("legal_risk") else "skipped",
            "playbook": playbook_status,
        }

        user_message = "일부 분석이 제한적으로 제공됩니다." if state.get("error_logs") else None

        spike_analysis = state.get("spike_analysis") or {}
        spike_nature = spike_analysis.get("spike_nature", "neutral")

        priority_map = {"urgent": 10, "high": 7, "medium": 5, "low": 3}
        severity_score = priority_map.get(playbook.get("priority", "low"), 5)

        sentiment_result = state.get("sentiment_result") or {}
        sentiment_shift = sentiment_result.get("sentiment_shift", "stable")
        trend_map = {"worsening": "escalating", "improving": "declining", "stable": "stable"}
        trend_direction = trend_map.get(sentiment_shift, "stable")

        spike_event = state.get("spike_event") or {}
        keyword = spike_event.get("keyword", "unknown")
        summary = f"{keyword} - {spike_nature} ?? ({playbook.get('situation_type', 'monitoring')})"

        state["executive_brief"] = {
            "summary": summary,
            "severity_score": severity_score,
            "trend_direction": trend_direction,
            "issue_polarity": spike_nature if spike_nature in ["positive", "negative", "mixed"] else "mixed",
            "spike_summary": spike_summary,
            "sentiment_summary": sentiment_summary,
            "legal_summary": legal_summary,
            "action_summary": action_summary,
            "opportunity_summary": opportunity_summary,
            "analysis_status": analysis_status,
            "user_message": user_message,
            "generated_at": _utcnow_z(),
            "analysis_duration_seconds": round(duration, 2),
            "dashboard_url": state.get("dashboard_url"),
            "incident_url": state.get("incident_url"),
        }

        _update_node_insight(state, "exec_brief", "generated")

        bot_token = os.getenv("SLACK_BOT_TOKEN")
        channel_id = os.getenv("SLACK_CHANNEL_ID")
        if bot_token and channel_id:
            try:
                from src.integrations.slack import format_to_slack, send_to_slack

                logger.info("Slack 전송 중...")
                slack_message = format_to_slack(state)
                success = send_to_slack(slack_message)
                if success:
                    logger.info("Slack 전송 완료")
                else:
                    logger.warning("Slack 전송 실패")
                    _add_error_log(
                        state,
                        stage="exec_brief",
                        error_type="api_error",
                        message="slack send_to_slack returned False",
                        details={"channel_id": channel_id},
                    )
                    _raise_if_hard_fail(state, "exec_brief", "slack send failed")
            except Exception as e:
                logger.warning(f"Slack 전송 실패: {e}")
                _add_error_log(
                    state,
                    stage="exec_brief",
                    error_type="exception",
                    message=f"slack send exception: {e}",
                    details={"channel_id": channel_id},
                )
                _raise_if_hard_fail(state, "exec_brief", "slack send exception", e)
        elif bot_token and not channel_id:
            logger.warning("SLACK CHANNEL ID와 토큰을 확인하세요.")
            _add_error_log(
                state,
                stage="exec_brief",
                error_type="schema_error",
                message="SLACK_BOT_TOKEN exists but SLACK_CHANNEL_ID missing",
            )
            _raise_if_hard_fail(state, "exec_brief", "slack channel id missing")

        return state

    except Exception as e:
        _add_error_log(state, "exec_brief", "exception", f"ExecBrief ?? ??: {str(e)}")
        state["executive_brief"] = {
            "summary": "분석 실패",
            "severity_score": 5,
            "trend_direction": "stable",
            "issue_polarity": "mixed",
            "spike_summary": None,
            "sentiment_summary": None,
            "legal_summary": None,
            "action_summary": None,
            "opportunity_summary": None,
            "analysis_status": {
                "spike_analyzer": "failed",
                "sentiment": "failed",
                "causality": "failed",
                "legal_rag": "failed",
                "playbook": "failed",
            },
            "user_message": "분석 중 오류가 발생했습니다.",
            "generated_at": _utcnow_z(),
            "analysis_duration_seconds": 0.0,
        }
        _update_node_insight(state, "exec_brief", "failed")
        _raise_if_hard_fail(state, "exec_brief", "exec brief generation failed", e)
        return state


# ============================================================
# ExecBrief 헬퍼
# ============================================================

def _generate_spike_summary(state: AnalysisState) -> Optional[str]:
    spike = state.get("spike_analysis")
    if not spike:
        return None
    warning = f", {spike['partial_data_warning']}" if spike.get("partial_data_warning") else ""
    return (
        f"{spike['spike_rate']}배 급등, "
        f"{spike['spike_type']} 타입, "
        f"{'Breakout 포함' if spike['viral_indicators']['has_breakout'] else '일반 급등'}"
        f"{warning}"
    )


def _generate_sentiment_summary(state: AnalysisState) -> Optional[str]:
    sentiment = state.get("sentiment_result")
    if not sentiment:
        return None
    dist = sentiment["sentiment_distribution"]
    dominant = sentiment["dominant_sentiment"]
    return f"{dominant} {dist[dominant]*100:.0f}%, 분석 {sentiment['analyzed_count']}건"


def _generate_legal_summary(state: AnalysisState) -> str:
    legal = state.get("legal_risk")
    if not legal:
        return "법률 검토 미수행"
    if legal["clearance_status"] == "clear":
        return "법적 리스크 없음"
    if legal["clearance_status"] == "review_needed":
        return f"법적 검토 필요 ({legal['overall_risk_level']} 리스크)"
    return f"법적 리스크 높음 ({legal['overall_risk_level']})"


def _generate_action_summary(state: AnalysisState) -> Optional[str]:
    playbook = state.get("playbook")
    if not playbook or not playbook.get("recommended_actions"):
        return None
    primary_action = playbook["recommended_actions"][0]
    return f"{primary_action['description']} ({primary_action['urgency']} 우선순위)"


def _generate_opportunity_summary(state: AnalysisState) -> Optional[str]:
    spike = state.get("spike_analysis") or {}
    if spike.get("spike_nature") != "positive":
        return None
    playbook = state.get("playbook") or {}
    opportunities = playbook.get("key_opportunities", [])
    if not opportunities:
        return None
    return ", ".join(opportunities[:2])

