# -*- coding: utf-8 -*-

"""
DOLPIN 프롬프트 중앙화 모듈

모든 Agent의 프롬프트와 응답 파싱 로직을 중앙에서 관리합니다.
각 agent에서는 이 모듈의 함수를 import하여 사용합니다.
"""

# ============================================================
# PlaybookAgent 프롬프트
# ============================================================
def build_playbook_enhancement_prompt(
    state: dict,
    base_strategy: dict
) -> str:
    """
    PlaybookAgent LLM Enhancement 프롬프트 생성
    
    룰 베이스로 생성한 전략 구조에 자연어 설명을 추가하기 위한 프롬프트
    
    Args:
        state: AnalysisState (전체 분석 상태)
        base_strategy: 룰 베이스로 생성한 기본 전략
    
    Returns:
        str: LLM enhancement 프롬프트
    """
    
    # State에서 컨텍스트 추출
    spike_event = state.get("spike_event", {})
    spike = state.get("spike_analysis", {})
    sentiment = state.get("sentiment_result", {})
    legal = state.get("legal_risk", {})
    causality = state.get("causality_result", {})
    
    keyword = spike_event.get("keyword", "미상")
    spike_rate = spike.get("spike_rate", 0)
    spike_nature = spike.get("spike_nature", "neutral")
    dominant = sentiment.get("dominant_sentiment", "neutral") if sentiment else "neutral"
    
    # 팬 감정 비율
    dist = sentiment.get("sentiment_distribution", {}) if sentiment else {}
    positive_pct = (dist.get("support", 0) + dist.get("excitement", 0)) * 100
    negative_pct = (
        dist.get("concern", 0) + 
        dist.get("criticism", 0) + 
        dist.get("anger", 0) + 
        dist.get("disappointment", 0)
    ) * 100
    
    # 상황 타입별 맥락
    situation_type = base_strategy["situation_type"]
    situation_context_map = {
        "crisis": "부정적 이슈로 팬들의 불안과 비판이 커지는 위기 상황입니다.",
        "opportunity": "긍정적 바이럴로 팬덤 확장 기회가 있는 상황입니다.",
        "monitoring": "일반적인 팬 반응으로 특별한 대응이 필요 없는 상황입니다."
    }
    situation_context = situation_context_map.get(situation_type, "현재 상황")
    
    # 추가 컨텍스트
    legal_context = ""
    if legal:
        legal_context = f"""
- 법적 리스크: {legal.get('overall_risk_level', '알 수 없음')}
- 검토 상태: {legal.get('clearance_status', '알 수 없음')}"""
    
    causality_context = ""
    if causality:
        causality_context = f"""
- 확산 패턴: {causality.get('cascade_pattern', '알 수 없음')}
- 트리거 소스: {causality.get('trigger_source', '알 수 없음')}"""
    
    # 액션 리스트 생성
    actions = base_strategy["recommended_actions"]
    action_list = []
    
    action_names = {
        "legal_response": "법률팀 검토 및 대응",
        "issue_statement": "공식 입장문 발표",
        "internal_review": "내부 조사 및 재발 방지",
        "amplify_viral": "긍정 바이럴 확산",
        "engage_influencers": "허브 계정과 협력",
        "monitor_only": "팬 반응 모니터링",
        "prepare_communication": "팬 소통 준비"
    }
    
    for i, action in enumerate(actions, 1):
        action_type = action["action"]
        urgency = action["urgency"]
        action_name = action_names.get(action_type, action_type)
        action_list.append(f"{i}. {action_name} (긴급도: {urgency})")
    
    actions_text = "\n".join(action_list)
    
    # 프롬프트 조립
    prompt = f"""당신은 K-POP 팬덤 이슈 관리 및 PR 전략을 전문으로 하는 위기 관리 AI 어드바이저입니다.

[현재 상황]
- 아티스트: {keyword}
- 이슈 유형: {spike_rate:.1f}배 급등 ({spike_nature})
- 팬 감정: {dominant} 우세 (긍정 {positive_pct:.0f}%, 부정 {negative_pct:.0f}%)
- 상황 평가: {situation_context}
{legal_context}
{causality_context}

[대응 액션 리스트]
{actions_text}

[요청사항]
위 각 액션에 대해 다음을 한국어 존댓말로 작성해주세요:

1. **구체적 실행 방안** (2-3문장)
   - 무엇을, 어떻게, 누구를 대상으로 할 것인가
   - 실무적으로 실행 가능한 구체적인 방법 제시
   
2. **예상 효과 및 근거** (2-3문장)
   - 왜 이 액션이 필요한가
   - 어떤 효과를 기대하는가
   - 현재 상황에서 이 액션이 적합한 이유

3. **공식 성명 초안** (5-6문장, 필요시만)
   - 다음 액션일 때만 작성: "issue_statement", "amplify_viral"
   - crisis 상황 → 사과문 형태
   - opportunity 상황 → 감사/격려 메시지
   - monitoring 상황 → 입장문
   - 과도한 법적 표현 회피
   - 공감과 책임 인식이 드러나야 함
   - 간결하고 진정성 있는 톤

[출력 형식]
각 액션마다 아래 형식을 정확히 따라 작성하세요:

---ACTION_1_START---
구체적 실행 방안: [2-3문장으로 작성]
예상 효과 및 근거: [2-3문장으로 작성]
공식 성명 초안: [필요시만 5-6문장으로 작성, 불필요하면 생략]
---ACTION_1_END---

---ACTION_2_START---
구체적 실행 방안: [2-3문장으로 작성]
예상 효과 및 근거: [2-3문장으로 작성]
공식 성명 초안: [필요시만 5-6문장으로 작성, 불필요하면 생략]
---ACTION_2_END---

(계속...)
"""
    
    return prompt


def parse_playbook_enhancement_response(
    response_text: str,
    base_strategy: dict
) -> list:
    """
    PlaybookAgent LLM Enhancement 응답 파싱
    
    Args:
        response_text: LLM 응답 텍스트
        base_strategy: 룰 베이스로 생성한 기본 전략
    
    Returns:
        list: LLM으로 보강된 recommended_actions
    """
    
    actions = base_strategy["recommended_actions"]
    
    # 각 액션마다 LLM 생성 내용 추출
    for i, action in enumerate(actions, 1):
        start_marker = f"---ACTION_{i}_START---"
        end_marker = f"---ACTION_{i}_END---"
        
        start_idx = response_text.find(start_marker)
        end_idx = response_text.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            # 파싱 실패 시 기본값 유지
            action.setdefault("description", "대응 전략")
            action.setdefault("rationale", "")
            continue
        
        content = response_text[start_idx + len(start_marker):end_idx].strip()
        
        # description 추출
        if "구체적 실행 방안:" in content:
            desc_start = content.find("구체적 실행 방안:") + len("구체적 실행 방안:")
            desc_end = content.find("예상 효과 및 근거:")
            if desc_end != -1:
                action["description"] = content[desc_start:desc_end].strip()
            else:
                action["description"] = "대응 전략"
        else:
            action["description"] = "대응 전략"
        
        # rationale 추출
        if "예상 효과 및 근거:" in content:
            rat_start = content.find("예상 효과 및 근거:") + len("예상 효과 및 근거:")
            rat_end = content.find("공식 성명 초안:")
            if rat_end == -1:
                rat_end = len(content)
            action["rationale"] = content[rat_start:rat_end].strip()
        else:
            action["rationale"] = ""
        
        # draft 추출 (있으면)
        if "공식 성명 초안:" in content:
            draft_start = content.find("공식 성명 초안:") + len("공식 성명 초안:")
            draft = content[draft_start:].strip()
            if draft and len(draft) > 20:
                action["draft"] = draft
            else:
                action["draft"] = None
        else:
            action["draft"] = None
    
    return actions


# ============================================================
# 추가될 프롬프트 
# ============================================================

# def build_sentiment_prompt(text: str) -> str:
#     """SentimentAgent 프롬프트"""
#     pass

# def build_legal_prompt(data: dict) -> str:
#     """LegalRAG 프롬프트"""
#     pass

# def build_causality_prompt(data: dict) -> str:
#     """CausalityAgent 프롬프트"""
#     pass

# def build_amplification_prompt(data: dict) -> str:
#     """AmplificationAgent 프롬프트"""
#     pass