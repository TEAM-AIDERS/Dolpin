"""
LangGraph 노드 래퍼 함수
각 에이전트를 호출하고 State를 업데이트합니다.

버전: v1.1 (260114)
참조: [SPEC] Router 분기 로직 + State 정리 + 함수 인터페이스 v1.1

개발 진행 방식:
1. 초기: 모든 노드는 Stub(더미 데이터)으로 구현
2. 에이전트 완성 후: Stub을 실제 에이전트 호출로 교체
"""

import logging
from datetime import datetime
from typing import Dict, Any

from .state import AnalysisState, ErrorLog
from .edges import (
    route_after_spike_analysis,
    route_after_sentiment,
    route_after_causality
)

# 에이전트 연결
from src.agents.sentiment_agent import build_agent as build_sentiment_agent

# 추후 에이전트 연결 시 import
# from ..agents import spike_analyzer, causality, legal_rag, amplification, playbook

logger = logging.getLogger(__name__)


# ============================================================
# 유틸리티 함수
# ============================================================

def _add_error_log(
    state: AnalysisState,
    stage: str,
    error_type: str,
    message: str,
    details: Dict[str, Any] = None
) -> None:
    """에러 로그 추가"""
    error_log: ErrorLog = {
        "stage": stage,
        "error_type": error_type,
        "message": message,
        "occurred_at": datetime.utcnow().isoformat() + "Z",
        "trace_id": state.get("trace_id", "unknown"),
        "details": details
    }
    state["error_logs"].append(error_log)
    logger.error(f"[{stage}] {error_type}: {message}", extra={"trace_id": state["trace_id"]})


def _update_node_insight(state: AnalysisState, node_name: str, insight: str) -> None:
    """노드 인사이트 업데이트"""
    if "node_insights" not in state:
        state["node_insights"] = {}
    state["node_insights"][node_name] = insight
    logger.info(f"[{node_name}] Insight: {insight}", extra={"trace_id": state["trace_id"]})


# ============================================================
# 노드: SpikeAnalyzer
# ============================================================

def spike_analyzer_node(state: AnalysisState) -> AnalysisState:
    """
    급등 분석 노드
    
    TODO: spike_analyzer.analyze() 완성 후 교체
    """
    try:
        # ============================================================
        # TODO: 실제 에이전트 연결 시 아래 주석 해제
        # ============================================================
        # from ..agents import spike_analyzer
        # spike_event = state["spike_event"]
        # result = spike_analyzer.analyze(spike_event)
        
        # ============================================================
        # Stub: 더미 데이터 (개발 초기용)
        # ============================================================
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
            "reason": "급등 징후 분석 완료: 3.5배 급등율, 긍정적 바이럴, 높은 신뢰도(0.85)",
            "viral_indicators": {
                "is_trending": True,
                "has_breakout": True,
                "max_rise_rate": "Breakout",
                "breakout_queries": ["뉴진스 컴백"],
                "cross_platform": ["twitter", "google_trends"],
                "international_reach": 0.3
            }
        }
        
        # State 업데이트
        state["spike_analysis"] = result
        
        # Insight 생성
        insight = (
            f"{result['spike_rate']}배 급등, {result['spike_nature']} 바이럴"
            + (f", {result['partial_data_warning']}" if result['partial_data_warning'] else "")
        )
        _update_node_insight(state, "SpikeAnalyzer", insight)
        
        # spike 판단 근거를 node_insights에 저장
        state["node_insights"]["spike"] = result.get("reason", "급등 징후 분석 완료")
        
        logger.info(f"SpikeAnalyzer 완료: is_significant={result['is_significant']}")
        
    except Exception as e:
        _add_error_log(
            state,
            stage="spike_analyzer",
            error_type="exception",
            message=str(e),
            details={"keyword": state["spike_event"]["keyword"]}
        )
        # 실패 시에도 기본값 설정 (워크플로우 계속 진행)
        state["spike_analysis"] = {
            "is_significant": False,
            "spike_rate": 0.0,
            "spike_type": "organic",
            "spike_nature": "neutral",
            "peak_timestamp": datetime.utcnow().isoformat() + "Z",
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
        # 에러 발생 시 spike 판단 근거 저장
        state["node_insights"]["spike"] = "급등 분석 중 오류가 발생했습니다."
    
    return state


# ============================================================
# 노드: Router 1차
# ============================================================

def router1_node(state: AnalysisState) -> AnalysisState:
    """
    Router 1차: Skip vs Analyze
    """
    try:
        decision = route_after_spike_analysis(state)
        state["route1_decision"] = decision
        
        if decision == "skip":
            state["skipped"] = True
            state["skip_reason"] = "not_significant"
            logger.info("Router 1차: skip (is_significant=False)")
        else:
            state["skipped"] = False
            logger.info("Router 1차: analyze (is_significant=True)")
        
    except Exception as e:
        _add_error_log(state, "spike_analyzer", "exception", f"Router 1차 에러: {str(e)}")
        state["route1_decision"] = "skip"
        state["skipped"] = True
        state["skip_reason"] = "not_significant"
    
    return state


# ============================================================
# 노드: LexiconLookup
# ============================================================

def lexicon_lookup_node(state: AnalysisState) -> AnalysisState:
    """
    렉시콘 매칭 노드
    
    spike_event의 메시지들을 렉시콘과 매칭하여
    state["lexicon_matches"]에 결과 저장
    
    MCPClient Singleton을 통해 CSV 렉시콘 분석 수행
    """
    from src.server.mcp_client import MCPClient
    
    try:
        spike_event = state.get("spike_event")
        
        if not spike_event or "messages" not in spike_event:
            # 메시지가 없으면 빈 결과 반환
            state["lexicon_matches"] = None
            logger.info("LexiconLookup: 분석할 메시지 없음")
            return state
        
        # ============================================================
        # MCPClient를 통한 실제 렉시콘 분석
        # ============================================================
        mcp = MCPClient("custom_lexicon.csv")
        
        # 메시지들을 순회하며 렉시콘 매칭
        lexicon_matches_result = {}
        messages = spike_event.get("messages", [])
        
        for msg in messages:
            text = msg.get("text", "")
            if not text:
                continue
            
            # MCPClient.lexicon_analyze()로 매칭 항목 추출
            matches, risks = mcp.lexicon_analyze(text)
            
            # type별로 count 집계
            for match in matches:
                type_name = match.get("type", "unknown")
                if type_name not in lexicon_matches_result:
                    lexicon_matches_result[type_name] = {
                        "count": 0,
                        "type": type_name,
                        "terms": []
                    }
                lexicon_matches_result[type_name]["count"] += 1
                # 매칭된 실제 항목도 기록
                if "term" in match and match["term"] not in lexicon_matches_result[type_name]["terms"]:
                    lexicon_matches_result[type_name]["terms"].append(match["term"])
        
        # State 업데이트
        state["lexicon_matches"] = lexicon_matches_result if lexicon_matches_result else None
        
        # Node Insight 생성
        if lexicon_matches_result:
            top_types = sorted(
                lexicon_matches_result.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:3]
            
            types_str = ", ".join([f"{t[0]}({t[1]['count']})" for t in top_types])
            insight = f"렉시콘 매칭: {types_str}"
        else:
            insight = "렉시콘 매칭 결과 없음"
        
        _update_node_insight(state, "LexiconLookup", insight)
        
        logger.info(f"LexiconLookup 완료: {len(lexicon_matches_result)} 타입 매칭")
        
    except Exception as e:
        _add_error_log(
            state,
            stage="lexicon_lookup",
            error_type="exception",
            message=str(e),
            details={"keyword": state.get("spike_event", {}).get("keyword")}
        )
        # 실패 시 None 처리 (워크플로우 계속 진행)
        state["lexicon_matches"] = None
        logger.warning(f"LexiconLookup 실패: {str(e)}")
    
    return state


# ============================================================
# 노드: SentimentAgent
# ============================================================

def sentiment_node(state: AnalysisState) -> AnalysisState:
    """
    감정 분석 노드
    
    SentimentAgent를 사용하여 spike_event의 메시지들을 분석하고
    sentiment_result를 생성합니다.
    """
    try:
        spike_event = state.get("spike_event")
        
        if not spike_event or "messages" not in spike_event:
            logger.warning("No messages in spike_event for sentiment analysis")
            state["sentiment_result"] = None
            return state
        
        # 메시지 전처리 (리스트를 하나의 텍스트로 결합)
        messages = spike_event.get("messages", [])
        if isinstance(messages, list):
            combined_text = " ".join(str(m) for m in messages)
        else:
            combined_text = str(messages)
        
        # SentimentAgent 초기화 (Singleton으로 캐싱됨)
        # 첫 호출: 모델 로드, 이후 호출: 캐시 사용
        MODEL_PATH = state.get("sentiment_model_path", "models/sentiment_model")
        LEXICON_PATH = state.get("lexicon_path", "custom_lexicon.csv")
        DEVICE = state.get("device", "cpu")
        
        try:
            agent = build_sentiment_agent(MODEL_PATH, LEXICON_PATH, DEVICE)
        except Exception as e:
            logger.error(f"SentimentAgent 초기화 실패: {e}")
            # 모델 없으면 None 처리 (graceful fallback)
            state["sentiment_result"] = None
            return state
        
        # 감정 분석 실행
        analyzed_count = len(messages) if isinstance(messages, list) else 1
        result, route_meta = agent.analyze(combined_text, analyzed_count=analyzed_count)
        
        # state에 필요한 필드 추가
        result["analyzed_count"] = analyzed_count
        result["confidence"] = result.get("confidence", 0.5)
        result["sentiment_shift"] = "stable"  # TODO: 이전 상태와 비교하여 계산
        result["representative_messages"] = {
            result.get("dominant_sentiment", "support"): messages[:3] if isinstance(messages, list) else [combined_text[:50]]
        }
        
        # State 업데이트
        state["sentiment_result"] = result
        
        # Insight 생성
        dominant = result.get("dominant_sentiment", "unknown")
        confidence = result.get("confidence", 0)
        insight = f"{dominant} {confidence*100:.0f}% (신뢰도), {analyzed_count}건 분석"
        _update_node_insight(state, "Sentiment", insight)
        
        logger.info(
            f"Sentiment 완료: dominant={dominant}, confidence={confidence:.2f}, "
            f"route={route_meta.get('route', 'unknown')}"
        )
        
    except Exception as e:
        _add_error_log(
            state,
            stage="sentiment",
            error_type="exception",
            message=str(e),
            details={"spike_event_keyword": state.get("spike_event", {}).get("keyword")}
        )
        # 실패 시 None 처리
        state["sentiment_result"] = None
    
    return state


# ============================================================
# 노드: Router 2차
# ============================================================

def router2_node(state: AnalysisState) -> AnalysisState:
    """
    Router 2차: Sentiment Only vs Full Analysis
    
    Note: positive_viral_detected도 여기서 설정 (Router 3차에서 재사용)
    """
    try:
        spike_analysis = state.get("spike_analysis")
        sentiment_result = state.get("sentiment_result")
        
        if not spike_analysis or not sentiment_result:
            state["route2_decision"] = "sentiment_only"
            state["positive_viral_detected"] = False
            return state
        
        # positive_viral_detected 먼저 계산 (Router 3차에서 재사용)
        from .edges import _check_opportunity_signals
        has_opportunity = _check_opportunity_signals(spike_analysis, sentiment_result)
        state["positive_viral_detected"] = has_opportunity
        
        # 그 다음 decision 결정
        decision = route_after_sentiment(state)
        state["route2_decision"] = decision
        
        logger.info(
            f"Router 2차: {decision}, "
            f"positive_viral={has_opportunity}, "
            f"actionability={spike_analysis.get('actionability_score')}"
        )
        
    except Exception as e:
        _add_error_log(state, "sentiment", "exception", f"Router 2차 에러: {str(e)}")
        state["route2_decision"] = "sentiment_only"
        state["positive_viral_detected"] = False
    
    return state


# ============================================================
# 노드: CausalityAgent
# ============================================================

def causality_node(state: AnalysisState) -> AnalysisState:
    """
    인과관계 분석 노드
    
    TODO: causality.analyze_network() 완성 후 교체
    """
    try:
        # ============================================================
        # TODO: 실제 에이전트 연결 시 아래 주석 해제
        # ============================================================
        # from ..agents import causality
        # spike_event = state["spike_event"]
        # sentiment_context = state.get("sentiment_result")
        # result = causality.analyze_network(spike_event, sentiment_context)
        
        # ============================================================
        # Stub: 더미 데이터 (개발 초기용)
        # ============================================================
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
            "retweet_network_metrics": {
                "centralization": 0.7,
                "avg_degree": 15.3
            },
            "cascade_pattern": "viral",
            "estimated_origin_time": "2026-01-10T09:30:00Z",
            "key_propagation_paths": [
                "인플루언서A → 팬계정B → 일반유저들"
            ]
        }
        
        # State 업데이트
        state["causality_result"] = result
        
        # Insight 생성
        insight = (
            f"{result['trigger_source']} 주도, "
            f"{result['cascade_pattern']} 패턴, "
            f"허브 계정 {len(result['hub_accounts'])}개"
        )
        _update_node_insight(state, "Causality", insight)
        
        logger.info(f"Causality 완료: trigger={result['trigger_source']}")
        
    except Exception as e:
        _add_error_log(state, "causality", "exception", str(e))
        state["causality_result"] = None
    
    return state


# ============================================================
# 노드: Router 3차
# ============================================================

def router3_node(state: AnalysisState) -> AnalysisState:
    """
    Router 3차: Legal vs Amplification
    """
    try:
        decision = route_after_causality(state)
        state["route3_decision"] = decision
        
        logger.info(f"Router 3차: {decision}")
        
    except Exception as e:
        _add_error_log(state, "causality", "exception", f"Router 3차 에러: {str(e)}")
        # 에러 시 안전하게 legal 경로로
        state["route3_decision"] = "legal"
    
    return state


# ============================================================
# 노드: LegalRAGAgent
# ============================================================

def legal_rag_node(state: AnalysisState) -> AnalysisState:
    """
    법률 리스크 검토 노드
    
    TODO: legal_rag.check_legal_risk() 완성 후 교체
    """
    try:
        # ============================================================
        # TODO: 실제 에이전트 연결 시 아래 주석 해제
        # ============================================================
        # from ..agents import legal_rag
        # from ..langgraph.state import LegalRAGInput
        #
        # spike_event = state["spike_event"]
        # spike_analysis = state["spike_analysis"]
        # sentiment_result = state["sentiment_result"]
        #
        # query_context: LegalRAGInput = {
        #     "messages": [msg["text"] for msg in spike_event["messages"]],
        #     "spike_nature": spike_analysis["spike_nature"],
        #     "dominant_sentiment": sentiment_result["dominant_sentiment"],
        #     "keyword": spike_event["keyword"],
        #     "spike_rate": spike_analysis["spike_rate"],
        #     "fanwar_targets": sentiment_result.get("fanwar_targets")
        # }
        # result = legal_rag.check_legal_risk(query_context)
        
        # ============================================================
        # Stub: 더미 데이터 (개발 초기용)
        # ============================================================
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
        
        # Insight 생성
        insight = f"법적 리스크 {result['overall_risk_level']}, {result['clearance_status']}"
        _update_node_insight(state, "LegalRAG", insight)
        
        logger.info(f"LegalRAG 완료: risk={result['overall_risk_level']}, RAG수행={result['rag_performed']}")
        
    except Exception as e:
        _add_error_log(state, "legal_rag", "exception", str(e))
        # 실패 시 안전 모드 (review_needed)
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
    
    return state


# ============================================================
# 노드: Amplification
# ============================================================

def amplification_node(state: AnalysisState) -> AnalysisState:
    """
    긍정 바이럴 기회 요약 노드
    
    TODO: amplification.summarize_opportunity() 완성 후 교체
    """
    try:
        # ============================================================
        # TODO: 실제 에이전트 연결 시 아래 주석 해제
        # ============================================================
        # from ..agents import amplification
        # result = amplification.summarize_opportunity(state)
        
        # ============================================================
        # Stub: 더미 데이터 (개발 초기용)
        # ============================================================
        causality = state.get("causality_result", {})
        sentiment = state.get("sentiment_result", {})
        spike = state.get("spike_analysis", {})
        
        result = {
            "top_platforms": spike.get("viral_indicators", {}).get("cross_platform", ["twitter"]),
            "hub_accounts": [
                {
                    "account_id": hub["account_id"],
                    "influence_score": hub["influence_score"],
                    "suggested_action": "공식 콘텐츠 제공"
                }
                for hub in causality.get("hub_accounts", [])[:3]
            ],
            "representative_messages": [
                {"text": msg, "engagement": 1000}
                for msg in sentiment.get("representative_messages", {}).get("support", [])[:3]
            ],
            "suggested_actions": [
                "official_twitter 참여 독려",
                "챌린지 공식 안무 영상 공유",
                "긍정 콘텐츠 리트윗"
            ]
        }
        
        # State 업데이트
        state["amplification_summary"] = result
        
        # Insight 생성
        insight = (
            f"바이럴 기회 감지, "
            f"플랫폼 {len(result['top_platforms'])}개, "
            f"주요 계정 {len(result['hub_accounts'])}개"
        )
        _update_node_insight(state, "Amplification", insight)
        
        logger.info(f"Amplification 완료: 플랫폼={result['top_platforms']}")
        
    except Exception as e:
        _add_error_log(state, "causality", "exception", f"Amplification 에러: {str(e)}")
        state["amplification_summary"] = None
    
    return state


# ============================================================
# 노드: PlaybookAgent
# ============================================================

def playbook_node(state: AnalysisState) -> AnalysisState:
    """대응 전략 생성 노드"""
    try:
        from src.agents.playbook_agent import generate_strategy
        
        playbook_result = generate_strategy(state, use_llm_enhancement=True)
        state["playbook"] = playbook_result
        
        # Insight 생성
        situation_type = playbook_result.get("situation_type", "unknown")
        priority = playbook_result.get("priority", "unknown")
        action_count = len(playbook_result.get("recommended_actions", []))
        
        insight = f"{situation_type} 전략, {priority} 우선순위, {action_count}개 액션"
        _update_node_insight(state, "Playbook", insight)
        
        logger.info(f"Playbook 완료: {situation_type}, {action_count} 액션")
        
    except Exception as e:
        _add_error_log(state, "playbook", "exception", str(e))
        state["playbook"] = None
    
    return state 


# ============================================================
# 노드: ExecBrief
# ============================================================

def exec_brief_node(state: AnalysisState) -> AnalysisState:
    """
    임원 브리핑 생성 노드
    
    모든 분석 결과를 종합하여 최종 출력 생성
    """
    try:
        # 분석 시작 시간
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
        
        # 에러 메시지 생성
        user_message = None
        if state.get("error_logs"):
            user_message = "일부 분석이 제한적으로 제공됩니다."
        
        # Issue polarity 판단
        spike_analysis = state.get("spike_analysis", {})
        spike_nature = spike_analysis.get("spike_nature", "neutral")
        
        # Severity score 계산
        playbook = state.get("playbook", {})
        priority_map = {"urgent": 10, "high": 7, "medium": 5, "low": 3}
        severity_score = priority_map.get(playbook.get("priority", "low"), 5)
        
        # Trend direction 판단
        sentiment_result = state.get("sentiment_result", {})
        sentiment_shift = sentiment_result.get("sentiment_shift", "stable")
        trend_map = {"worsening": "escalating", "improving": "declining", "stable": "stable"}
        trend_direction = trend_map.get(sentiment_shift, "stable")
        
        # 전체 요약 생성
        keyword = state["spike_event"]["keyword"]
        summary = f"{keyword} - {spike_nature} 이슈 ({playbook.get('situation_type', 'monitoring')})"
        
        # ExecBrief 생성
        exec_brief = {
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
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "analysis_duration_seconds": round(duration, 2)
        }
        
        state["executive_brief"] = exec_brief
        
        logger.info(f"ExecBrief 생성 완료: {summary}")
        
    except Exception as e:
        _add_error_log(state, "playbook", "exception", f"ExecBrief 생성 에러: {str(e)}")
        # 최소한의 브리핑이라도 생성
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
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "analysis_duration_seconds": 0.0
        }
    
    return state


# ============================================================
# ExecBrief 헬퍼 함수
# ============================================================

def _generate_spike_summary(state: AnalysisState) -> str:
    """Spike 요약 생성"""
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


def _generate_sentiment_summary(state: AnalysisState) -> str:
    """Sentiment 요약 생성"""
    sentiment = state.get("sentiment_result")
    if not sentiment:
        return None
    
    dist = sentiment["sentiment_distribution"]
    dominant = sentiment["dominant_sentiment"]
    
    return f"{dominant} {dist[dominant]*100:.0f}%, 분석 {sentiment['analyzed_count']}건"


def _generate_legal_summary(state: AnalysisState) -> str:
    """Legal 요약 생성"""
    legal = state.get("legal_risk")
    if not legal:
        return "법률 검토 미수행"
    
    if legal["clearance_status"] == "clear":
        return "법적 리스크 없음"
    elif legal["clearance_status"] == "review_needed":
        return f"법적 검토 필요 ({legal['overall_risk_level']} 리스크)"
    else:
        return f"법적 리스크 높음 ({legal['overall_risk_level']})"


def _generate_action_summary(state: AnalysisState) -> str:
    """Action 요약 생성"""
    playbook = state.get("playbook")
    if not playbook or not playbook.get("recommended_actions"):
        return None
    
    actions = playbook["recommended_actions"]
    primary_action = actions[0]
    
    return f"{primary_action['description']} ({primary_action['urgency']} 우선순위)"


def _generate_opportunity_summary(state: AnalysisState) -> str:
    """Opportunity 요약 생성 (긍정 이슈만)"""
    spike = state.get("spike_analysis", {})
    if spike.get("spike_nature") != "positive":
        return None
    
    playbook = state.get("playbook", {})
    opportunities = playbook.get("key_opportunities", [])
    
    if not opportunities:
        return None
    
    return ", ".join(opportunities[:2])