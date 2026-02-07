"""
LangGraph 노드 래퍼 함수
각 에이전트를 호출하고 State를 업데이트합니다.

버전: v2 (260203)
- state 저장 형태 확장 반영
- node_insights 키를 graph node id와 동일하게 통일
- lexicon_lookup는 MCPClient singleton(get_mcp_client) 사용
"""

import logging, os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .state import AnalysisState, ErrorLog
from .edges import (
    route_after_spike_analysis,
    route_after_sentiment,
    route_after_causality
)

from src.agents.sentiment_agent import build_agent as build_sentiment_agent

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



# ============================================================
# spike_analyzer
# ============================================================

def spike_analyzer_node(state: AnalysisState) -> AnalysisState:
    """
    급등 분석 노드 (현재 stub)
    TODO: spike analyzer 실제 연결 시 교체
    """
    _ensure_state_collections(state)

    try:
        # Stub: 더미 데이터
        spike_event = state["spike_event"]
        result = {
            "is_significant": True,
            "spike_rate": 3.5,
            "spike_type": "organic",
            "spike_nature": "positive",
            "peak_timestamp": "2026-01-10T10:30:00Z",
            "duration_minutes": 60,
            "confidence": 0.85,
            "actionability_score": 0.7,
            "data_completeness": "confirmed",
            "partial_data_warning": None,
            "viral_indicators": {
                "is_trending": True,
                "has_breakout": True,
                "max_rise_rate": "Breakout",
                "breakout_queries": ["뉴진스 컴백"],
                "cross_platform": ["twitter", "google_trends"],
                "international_reach": 0.3
            }
        }

        state["spike_analysis"] = result

        insight = f"{result['spike_rate']}배 급등, {result['spike_nature']} 바이럴"
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


# ============================================================
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

        for msg in messages:
            text = (msg or {}).get("text", "")
            if not text:
                continue

            res = mcp.lexicon_analyze(text) or {}
            matches = res.get("matches", [])  # dict 리턴 기준

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

        MODEL_PATH = state.get("sentiment_model_path", "models/sentiment_model")
        LEXICON_PATH = state.get("lexicon_path", "custom_lexicon.csv")
        DEVICE = state.get("device", "cpu")

        try:
            agent = build_sentiment_agent(MODEL_PATH, LEXICON_PATH, DEVICE)
        except Exception as e:
            logger.error(f"SentimentAgent 초기화 실패: {e}")
            state["sentiment_result"] = None
            _update_node_insight(state, "sentiment", "모델 초기화 실패")
            return state

        # 감정 분석 실행
        analyzed_count = len(messages)
        result, route_meta = agent.analyze(combined_text, analyzed_count=analyzed_count)

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
    인과관계 분석 노드 (현재 stub)
    """
    _ensure_state_collections(state)

    try:
        result = {
            "trigger_source": "influencer",
            "hub_accounts": [
                {
                    "account_id": "user_abc",
                    "influence_score": 0.9,
                    "follower_count": 150000,
                    "account_type": "influencer"
                }
            ],
            "retweet_network_metrics": {"centralization": 0.7, "avg_degree": 15.3},
            "cascade_pattern": "viral",
            "estimated_origin_time": "2026-01-10T09:30:00Z",
            "key_propagation_paths": ["인플루언서A → 팬계정B → 일반유저들"]
        }

        state["causality_result"] = result
        _update_node_insight(state, "causality", f"{result['trigger_source']} 주도, {result['cascade_pattern']} 패턴")
        return state

    except Exception as e:
        _add_error_log(state, "causality", "exception", str(e))
        state["causality_result"] = None
        _update_node_insight(state, "causality", "인과 분석 실패(예외)")
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

def legal_rag_node(state: AnalysisState) -> AnalysisState:
    """
    법률 리스크 검토 노드 (현재 stub)
    """
    _ensure_state_collections(state)

    try:
        result = {
            "overall_risk_level": "low",
            "clearance_status": "clear",
            "confidence": 0.95,
            "rag_required": False,
            "rag_performed": False,
            "rag_confidence": None,
            "risk_assessment": None,
            "recommended_action": [],
            "referenced_documents": [],
            "signals": {
                "legal_keywords_detected": False,
                "matched_keywords": [],
                "reason": "none"
            }
        }

        # State 업데이트
        state["legal_risk"] = result
        _update_node_insight(state, "legal_rag", f"{result['overall_risk_level']}/{result['clearance_status']}")
        return state

    except Exception as e:
        _add_error_log(state, "legal_rag", "exception", str(e))
        state["legal_risk"] = {
            "overall_risk_level": "medium",
            "clearance_status": "review_needed",
            "confidence": 0.5,
            "rag_required": False,
            "rag_performed": False,
            "rag_confidence": None,
            "risk_assessment": None,
            "recommended_action": ["수동 검토 필요"],
            "referenced_documents": [],
            "signals": None
        }
        _update_node_insight(state, "legal_rag", "review_needed (exception)")
        return state


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
        
        # ===== Representative Messages (초간단!) =====
        support_msgs = (sentiment.get("representative_messages", {}).get("support") or [])
        representative_messages = [
            {"text": msg}  # 👈 text만!
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
# exec_brief
# ============================================================

def exec_brief_node(state: AnalysisState) -> AnalysisState:
    """브리핑 생성 노드"""
    _ensure_state_collections(state)

    try:
        workflow_start = datetime.fromisoformat(state["workflow_start_time"].replace("Z", "+00:00"))
        duration = (datetime.now(workflow_start.tzinfo) - workflow_start).total_seconds()

        # 각 섹션 요약 생성
        spike_summary = _generate_spike_summary(state)
        sentiment_summary = _generate_sentiment_summary(state)
        legal_summary = _generate_legal_summary(state)
        action_summary = _generate_action_summary(state)
        opportunity_summary = _generate_opportunity_summary(state)

        # 분석 상태 확인
        analysis_status = {
            "spike_analyzer": "success" if state.get("spike_analysis") else "failed",
            "sentiment": "success" if state.get("sentiment_result") else "failed",
            "causality": "success" if state.get("causality_result") else "skipped",
            "legal_rag": "success" if state.get("legal_risk") else "skipped",
            "playbook": "success" if state.get("playbook") else "failed"
        }

        user_message = "일부 분석이 제한적으로 제공됩니다." if state.get("error_logs") else None

        spike_analysis = state.get("spike_analysis") or {}
        spike_nature = spike_analysis.get("spike_nature", "neutral")

        # Severity score 계산
        playbook = state.get("playbook") or {}
        priority_map = {"urgent": 10, "high": 7, "medium": 5, "low": 3}
        severity_score = priority_map.get(playbook.get("priority", "low"), 5)

        sentiment_result = state.get("sentiment_result") or {}
        sentiment_shift = sentiment_result.get("sentiment_shift", "stable")
        trend_map = {"worsening": "escalating", "improving": "declining", "stable": "stable"}
        trend_direction = trend_map.get(sentiment_shift, "stable")

        keyword = state["spike_event"]["keyword"]
        summary = f"{keyword} - {spike_nature} 이슈 ({playbook.get('situation_type', 'monitoring')})"

        # ExecBrief 생성
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
            "analysis_duration_seconds": round(duration, 2)
        }

        _update_node_insight(state, "exec_brief", "generated")
        return state

    except Exception as e:
        _add_error_log(state, "exec_brief", "exception", f"ExecBrief 생성 에러: {str(e)}")
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
                "playbook": "failed"
            },
            "user_message": "분석 중 오류가 발생했습니다.",
            "generated_at": _utcnow_z(),
            "analysis_duration_seconds": 0.0
        }
        _update_node_insight(state, "exec_brief", "failed")
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
