"""
LangGraph Router 분기 로직
DOLPIN 워크플로우의 조건부 분기를 처리합니다.

버전: v1.1 (260114)
참조: [SPEC] Router 분기 로직 + State 정리 + 함수 인터페이스 v1.1
"""

from typing import Literal
from .state import AnalysisState


# ============================================================
# Router 1차: Skip vs Analyze
# ============================================================

def route_after_spike_analysis(state: AnalysisState) -> Literal["skip", "analyze"]:
    """
    급등 분석 후 분기
    
    조건:
    - is_significant == False → skip (로그만 기록, 종료)
    - is_significant == True → analyze (Sentiment로 진행)
    
    Args:
        state: 전체 분석 상태
    
    Returns:
        "skip" | "analyze"
    """
    spike_analysis = state.get("spike_analysis")
    
    if not spike_analysis:
        # 예외 상황: spike_analysis가 없으면 skip
        return "skip"
    
    if not spike_analysis["is_significant"]:
        return "skip"
    
    return "analyze"


# ============================================================
# Router 2차: Sentiment Only vs Full Analysis
# ============================================================

def route_after_sentiment(state: AnalysisState) -> Literal["sentiment_only", "full_analysis"]:
    """
    감정 분석 후 분기
    
    분기 로직 (3단계):
    
    1단계: 낮은 actionability_score
    - actionability_score < 0.3 → sentiment_only
    
    2단계: 높은 actionability_score
    - actionability_score >= 0.6 → full_analysis
    
    3단계: 중간 actionability_score (0.3 ~ 0.6)
    - 위기 신호 또는 기회 신호 확인
    - 신호 있음 → full_analysis
    - 신호 없음 → sentiment_only
    
    Args:
        state: 전체 분석 상태
    
    Returns:
        "sentiment_only" | "full_analysis"
    """
    spike_analysis = state.get("spike_analysis")
    sentiment_result = state.get("sentiment_result")
    
    if not spike_analysis or not sentiment_result:
        # 예외 상황: 필수 데이터 없으면 sentiment_only로 안전하게 처리
        return "sentiment_only"
    
    actionability_score = spike_analysis["actionability_score"]
    
    # 1단계: 낮은 actionability → sentiment_only
    if actionability_score < 0.3:
        return "sentiment_only"
    
    # 2단계: 높은 actionability → full_analysis
    if actionability_score >= 0.6:
        return "full_analysis"
    
    # 3단계: 중간 actionability (0.3 ~ 0.6) → 위기/기회 신호 확인
    has_crisis_signal = _check_crisis_signals(sentiment_result)
    has_opportunity_signal = _check_opportunity_signals(spike_analysis, sentiment_result)
    
    # Note: positive_viral_detected는 router2_node()에서 설정됨
    
    if has_crisis_signal or has_opportunity_signal:
        return "full_analysis"
    
    return "sentiment_only"


def _check_crisis_signals(sentiment_result) -> bool:
    """
    위기 신호 확인
    
    조건 (하나라도 해당):
    - boycott >= 0.2
    - fanwar > 0.1
    - dominant_sentiment이 부정적 신호 (disappointment, boycott, fanwar)
    - has_mixed_sentiment + sentiment_shift == "worsening"
    """
    dist = sentiment_result["sentiment_distribution"]
    
    # boycott >= 0.2
    if dist.get("boycott", 0) >= 0.2:
        return True
    
    # fanwar > 0.1
    if dist.get("fanwar", 0) > 0.1:
        return True
    
    # dominant_sentiment이 부정적 신호
    dominant = sentiment_result.get("dominant_sentiment")
    if dominant in ["disappointment", "boycott", "fanwar"]:
        return True
    
    # has_mixed_sentiment + sentiment_shift == "worsening"
    if (sentiment_result.get("has_mixed_sentiment") and 
        sentiment_result.get("sentiment_shift") == "worsening"):
        return True
    
    return False


def _check_opportunity_signals(spike_analysis, sentiment_result) -> bool:
    """
    기회 신호 확인
    
    긍정 바이럴 조건 (모두 만족):
    - spike_nature == "positive"
    - spike_rate >= 3.0
    - support >= 0.5
    """
    # spike_nature == "positive"
    if spike_analysis.get("spike_nature") != "positive":
        return False
    
    # spike_rate >= 3.0
    if spike_analysis.get("spike_rate", 0) < 3.0:
        return False
    
    # support >= 0.5
    dist = sentiment_result["sentiment_distribution"]
    if dist.get("support", 0) < 0.5:
        return False
    
    return True


# ============================================================
# Router 3차: Legal vs Amplification
# ============================================================

def route_after_causality(state: AnalysisState) -> Literal["legal", "amplification"]:
    """
    인과관계 분석 후 분기
    
    조건:
    - positive_viral_detected == True → amplification (긍정 바이럴 확산)
    - 그 외 → legal (법적 리스크 체크)
    
    Args:
        state: 전체 분석 상태
    
    Returns:
        "legal" | "amplification"
    """
    positive_viral = state.get("positive_viral_detected", False)
    
    if positive_viral:
        return "amplification"
    
    return "legal"


# ============================================================
# 조건부 엣지 헬퍼
# ============================================================

def should_continue_after_router1(state: AnalysisState) -> Literal["sentiment", "end"]:
    """
    Router 1차 이후 계속 진행 여부
    
    Returns:
        "sentiment" - analyze 경로, Sentiment로 진행
        "end" - skip 경로, 종료
    """
    route1 = state.get("route1_decision")
    
    if route1 == "skip":
        return "end"
    
    return "sentiment"


def should_continue_after_router2(state: AnalysisState) -> Literal["playbook", "causality"]:
    """
    Router 2차 이후 계속 진행 여부
    
    Returns:
        "playbook" - sentiment_only 경로, Playbook으로 직행
        "causality" - full_analysis 경로, Causality로 진행
    """
    route2 = state.get("route2_decision")
    
    if route2 == "sentiment_only":
        return "playbook"
    
    return "causality"


def should_continue_after_router3(state: AnalysisState) -> Literal["legal_rag", "amplification"]:
    """
    Router 3차 이후 계속 진행 여부
    
    Returns:
        "amplification" - 긍정 바이럴 경로
        "legal_rag" - 법적 리스크 체크 경로
    """
    route3 = state.get("route3_decision")
    
    if route3 == "amplification":
        return "amplification"
    
    return "legal_rag"