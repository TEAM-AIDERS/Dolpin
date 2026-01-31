# -*- coding: utf-8 -*-
"""
PlaybookAgent - Hybrid Strategy Generator
룰 베이스 구조 + LLM 자연어 생성
"""

import os
from typing import Dict, Any, Optional
from openai import OpenAI
import logging

from src.dolpin_langgraph.state import AnalysisState
from src.prompt import (
    build_playbook_enhancement_prompt,
    parse_playbook_enhancement_response
)

logger = logging.getLogger(__name__)


def generate_strategy(
    state: AnalysisState,
    use_llm_enhancement: bool = True
) -> Dict[str, Any]:
    """
    대응 전략 생성 (하이브리드)
    
    Args:
        state: 전체 분석 상태
        use_llm_enhancement: True면 LLM으로 description, rationale, draft 생성
    
    Returns:
        dict: PlaybookResult
    """
    
    route2 = state.get("route2_decision")
    route3 = state.get("route3_decision")
    
    # ===== 1. 룰 베이스로 기본 구조 생성 =====
    if route2 == "sentiment_only":
        base_strategy = _generate_monitoring_strategy(state)
    elif route2 == "full_analysis":
        if route3 == "amplification":
            base_strategy = _generate_opportunity_strategy(state)
        else:  # legal
            base_strategy = _generate_crisis_strategy(state)
    else:
        base_strategy = _generate_default_strategy(state)
    
    # ===== 2. LLM으로 자연어 보강 (선택적) =====
    if use_llm_enhancement:
        enhanced_strategy = _enhance_with_llm(state, base_strategy)
        if enhanced_strategy:
            return enhanced_strategy
    
    return base_strategy


# ============================================================
# 룰 베이스 전략 생성 (기본 구조만)
# ============================================================

def _generate_crisis_strategy(state: AnalysisState) -> Dict[str, Any]:
    """위기 대응 전략 (Legal 경로) - 기본 구조만"""
    legal = state.get("legal_risk", {})
    sentiment = state.get("sentiment_result", {})
    spike = state.get("spike_analysis", {})
    
    risk_level = legal.get("overall_risk_level", "medium")
    clearance = legal.get("clearance_status", "review_needed")
    spike_rate = spike.get("spike_rate", 0)
    dominant = sentiment.get("dominant_sentiment", "neutral")
    
    # Priority 결정
    if risk_level == "critical" or clearance == "high_risk":
        priority = "urgent"
    elif risk_level == "high" or spike_rate >= 5.0:
        priority = "high"
    else:
        priority = "medium"
    
    # 액션 리스트 (구조만, 설명은 LLM이 채움)
    actions = []
    
    # 1. Legal 대응
    if clearance in ["review_needed", "high_risk"]:
        actions.append({
            "action": "legal_response",
            "urgency": "urgent" if clearance == "high_risk" else "high",
            "description": "",  # LLM이 채움
            "rationale": "",    # LLM이 채움
            "legal_basis": _extract_legal_basis(legal)
        })
    
    # 2. 공식 입장문
    if dominant in ["criticism", "anger", "disappointment", "concern"]:
        actions.append({
            "action": "issue_statement",
            "urgency": priority,
            "description": "",  # LLM이 채움
            "rationale": "",    # LLM이 채움
            "draft": None       # LLM이 생성
        })
    
    # 3. 내부 조사
    if spike_rate >= 3.0:
        actions.append({
            "action": "internal_review",
            "urgency": "high",
            "description": "",  # LLM이 채움
            "rationale": ""     # LLM이 채움
        })
    
    # Key Risks
    key_risks = []
    if legal.get("signals", {}).get("matched_keywords"):
        key_risks = legal["signals"]["matched_keywords"][:3]
    if spike.get("viral_indicators", {}).get("has_breakout"):
        key_risks.append("Breakout 급등 - 빠른 확산")
    
    return {
        "situation_type": "crisis",
        "priority": priority,
        "recommended_actions": actions,
        "key_risks": key_risks,
        "key_opportunities": [],
        "target_channels": ["press_release", "official_twitter", "fancafe"]
    }


def _generate_opportunity_strategy(state: AnalysisState) -> Dict[str, Any]:
    """기회 확대 전략 (Amplification 경로) - 기본 구조만"""
    amplification = state.get("amplification_summary", {})
    spike = state.get("spike_analysis", {})
    
    spike_rate = spike.get("spike_rate", 0)
    
    # Priority 결정
    if spike_rate >= 5.0:
        priority = "urgent"
    elif spike_rate >= 3.0:
        priority = "high"
    else:
        priority = "medium"
    
    # 액션 리스트
    actions = []
    
    # 1. 긍정 바이럴 확산
    rep_msgs = amplification.get("representative_messages", [])
    target_posts = []
    for i, msg in enumerate(rep_msgs[:3]):
        if isinstance(msg, dict):
            target_posts.append({
                "id": f"post-{i}",
                "source": "twitter",
                "source_message_id": msg.get("id", "unknown"),
                "url": f"https://twitter.com/status/{msg.get('id', '')}"
            })
    
    actions.append({
        "action": "amplify_viral",
        "urgency": priority,
        "description": "",      # LLM이 채움
        "rationale": "",        # LLM이 채움
        "target_posts": target_posts if target_posts else None,
        "draft": None           # LLM이 생성 (감사 메시지)
    })
    
    # 2. 허브 계정 협력
    if len(amplification.get("hub_accounts", [])) > 0:
        actions.append({
            "action": "engage_influencers",
            "urgency": "high",
            "description": "",  # LLM이 채움
            "rationale": ""     # LLM이 채움
        })
    
    # Key Opportunities
    key_opportunities = []
    hub_count = len(amplification.get("hub_accounts", []))
    if hub_count > 0:
        key_opportunities.append(f"{hub_count}개 허브 계정 활용 가능")
    
    platforms = amplification.get("top_platforms", [])
    if platforms:
        key_opportunities.append(f"{', '.join(platforms[:2])} 플랫폼 집중")
    
    viral_indicators = spike.get("viral_indicators", {})
    if viral_indicators.get("international_reach", 0) >= 0.3:
        key_opportunities.append("해외 팬덤 확장 기회")
    
    return {
        "situation_type": "opportunity",
        "priority": priority,
        "recommended_actions": actions,
        "key_risks": [],
        "key_opportunities": key_opportunities,
        "target_channels": platforms if platforms else ["official_twitter"]
    }


def _generate_monitoring_strategy(state: AnalysisState) -> Dict[str, Any]:
    """모니터링 전략 (sentiment_only 경로)"""
    spike = state.get("spike_analysis", {})
    sentiment = state.get("sentiment_result", {})
    
    spike_rate = spike.get("spike_rate", 0)
    dist = sentiment.get("sentiment_distribution", {}) if sentiment else {}
    
    priority = "medium" if spike_rate >= 3.0 else "low"
    
    actions = [{
        "action": "monitor_only",
        "urgency": priority,
        "description": "",  # LLM이 채움
        "rationale": ""     # LLM이 채움
    }]
    
    # 부정 감정 30% 이상이면 추가
    negative_pct = (
        dist.get("concern", 0) + 
        dist.get("criticism", 0) + 
        dist.get("anger", 0)
    )
    
    if negative_pct >= 0.3:
        actions.append({
            "action": "prepare_communication",
            "urgency": "medium",
            "description": "",  # LLM이 채움
            "rationale": ""     # LLM이 채움
        })
    
    return {
        "situation_type": "monitoring",
        "priority": priority,
        "recommended_actions": actions,
        "key_risks": [],
        "key_opportunities": [],
        "target_channels": ["fancafe", "official_twitter"]
    }


def _generate_default_strategy(state: AnalysisState) -> Dict[str, Any]:
    """기본 전략"""
    return {
        "situation_type": "monitoring",
        "priority": "low",
        "recommended_actions": [{
            "action": "monitor_only",
            "urgency": "low",
            "description": "상황 모니터링",
            "rationale": "분석 데이터 부족"
        }],
        "key_risks": [],
        "key_opportunities": [],
        "target_channels": ["fancafe"]
    }


# ============================================================
# LLM Enhancement (수정된 부분!)
# ============================================================

def _enhance_with_llm(
    state: AnalysisState,
    base_strategy: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    LLM으로 자연어 설명 생성
    
    Args:
        state: 전체 분석 상태
        base_strategy: 룰 베이스로 생성한 기본 전략
    
    Returns:
        dict: LLM으로 보강된 전략 (실패 시 None)
    """
    try:
        # OpenAI API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY 없음, LLM enhancement 스킵")
            return None
        
        client = OpenAI(api_key=api_key)
        
        # 프롬프트 생성 (src/prompt.py 사용)
        prompt = build_playbook_enhancement_prompt(state, base_strategy)
        
        # LLM 호출
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 K-POP 팬덤 이슈 관리 전문가입니다. 정확하고 실행 가능한 대응 전략을 제시합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        
        # 응답 파싱 및 병합 (src/prompt.py 사용)
        enhanced_actions = parse_playbook_enhancement_response(response_text, base_strategy)
        base_strategy["recommended_actions"] = enhanced_actions
        
        logger.info(f"LLM enhancement 완료: {len(enhanced_actions)}개 액션 보강")
        
        return base_strategy
        
    except Exception as e:
        logger.warning(f"LLM enhancement 실패: {e}")
        return None


# ============================================================
# 헬퍼 함수들
# ============================================================

def _extract_legal_basis(legal_risk: dict) -> Optional[str]:
    """법적 근거 추출"""
    if not legal_risk.get("referenced_documents"):
        return None
    
    docs = legal_risk["referenced_documents"]
    if docs and isinstance(docs, list) and len(docs) > 0:
        first_doc = docs[0]
        if isinstance(first_doc, dict):
            return first_doc.get("title", first_doc.get("summary", "관련 법률 조항"))
    
    return "관련 법률 조항"