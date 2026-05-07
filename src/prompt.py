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
K-POP 팬덤 문화에 대한 깊은 이해를 바탕으로 실무에서 실제로 효과 있는 전략을 제시합니다.

[K-POP 팬덤 문화 핵심 인사이트 — 반드시 반영]
- 팬 아트·팬 콘텐츠에 아티스트가 직접 피드백하거나 소개하는 것은 절대 금지 — 다른 팬들의 질투와 역반응을 유발함
- 긍정 바이럴 확산은 팬들이 자발 참여할 수 있는 트렌디한 챌린지 기획·촬영(릴스, 숏폼 포맷)이 가장 효과적
- 허브 계정 대응은 단순 협력·출연 섭외가 아니라, 팬들이 오랫동안 원해온 자체 컨텐츠(멤버 자체 예능, 브이로그, 비하인드 시리즈 등)를 직접 제작하여 허브 계정이 자연스럽게 언급·확산하도록 유도하는 방식
- 공식 소통은 장문 입장문보다 아티스트 본인의 짧고 진정성 있는 메시지(위버스·SNS 직접 소통)가 팬심에 더 효과적
- 팬들이 가장 원하는 것: 아티스트의 직접 소통, 비하인드 콘텐츠, 멤버들이 직접 만드는 자체 제작 예능/브이로그

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
   - 위 K-POP 팬덤 인사이트를 반영한 실무적으로 실행 가능한 구체적인 방법 제시

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
# ExecBrief LLM 내러티브 생성 프롬프트
# ============================================================

def build_exec_brief_prompt(state: dict) -> str:
    """
    분석 결과 전체를 받아 Slack 리포트용 내러티브 요약을 생성하는 프롬프트.
    spike_summary / sentiment_summary / opportunity_summary를 LLM이 자연어로 작성.
    """
    spike_event = state.get("spike_event") or {}
    spike       = state.get("spike_analysis") or {}
    sentiment   = state.get("sentiment_result") or {}
    causality   = state.get("causality_result") or {}
    playbook    = state.get("playbook") or {}

    keyword       = spike_event.get("keyword", "미상")
    spike_rate    = spike.get("spike_rate", 0)
    spike_nature  = spike.get("spike_nature", "neutral")
    spike_type    = spike.get("spike_type", "organic")
    situation     = playbook.get("situation_type", "monitoring")

    dominant      = sentiment.get("dominant_sentiment", "neutral")
    dist          = sentiment.get("sentiment_distribution") or {}
    analyzed      = sentiment.get("analyzed_count", 0)
    confidence    = sentiment.get("confidence", 0)

    # 대표 메시지 샘플 (최대 3건)
    rep_msgs = sentiment.get("representative_messages") or {}
    sample_lines = []
    for msgs in rep_msgs.values():
        for msg in (msgs or [])[:2]:
            if msg:
                sample_lines.append(f'- "{str(msg).strip()[:100]}"')
            if len(sample_lines) >= 3:
                break
        if len(sample_lines) >= 3:
            break
    sample_text = "\n".join(sample_lines) if sample_lines else "샘플 없음"

    # 확산 경로
    trigger  = causality.get("trigger_source", "알 수 없음")
    cascade  = causality.get("cascade_pattern", "알 수 없음")
    hub_cnt  = len(causality.get("hub_accounts") or [])

    situation_map = {
        "crisis":      "부정 이슈 — 팬들의 비판·실망이 확산 중인 위기 상황",
        "opportunity": "긍정 이슈 — 팬덤 확장 기회가 있는 바이럴 상황",
        "monitoring":  "일반 모니터링 — 특별한 대응이 필요 없는 안정적 상황",
    }
    situation_desc = situation_map.get(situation, situation)

    prompt = f"""당신은 K-POP 팬덤 이슈 관리를 전문으로 하는 PR 전략 AI입니다.
아래 분석 데이터를 바탕으로 Slack 리포트에 들어갈 자연어 요약 3가지를 작성하세요.

[분석 데이터]
- 아티스트/키워드: {keyword}
- 급등 규모: {spike_rate}배 ({spike_type} 유형, {spike_nature} 성격)
- 상황 유형: {situation_desc}
- 주요 팬 반응: {dominant} ({dist.get(dominant, 0)*100:.0f}%), 총 {analyzed}건 분석, 신뢰도 {confidence*100:.0f}%
- 감정 분포: {", ".join(f"{k} {v*100:.0f}%" for k, v in dist.items() if v > 0.05)}
- 확산 트리거: {trigger} / 패턴: {cascade} / 허브 계정: {hub_cnt}개
- 팬 반응 샘플:
{sample_text}

[요청]
아래 3가지를 각각 2~3문장의 한국어 존댓말로 작성하세요.
데이터 수치를 자연스럽게 녹여서 실무자가 바로 이해할 수 있게 작성하세요.

1. **현재 상황 요약** — 무슨 일이 일어나고 있는지, 얼마나 빠르게 확산되는지
2. **팬 반응 요약** — 팬들이 어떤 감정으로 무슨 이야기를 하는지
3. **확산 기회/위험** — 지금 상황에서 주목해야 할 점 (긍정이면 기회, 부정이면 위험)

[출력 형식] 반드시 아래 마커를 지켜 출력하세요.
---SPIKE_START---
(현재 상황 요약 2~3문장)
---SPIKE_END---
---SENTIMENT_START---
(팬 반응 요약 2~3문장)
---SENTIMENT_END---
---OPPORTUNITY_START---
(확산 기회/위험 2~3문장)
---OPPORTUNITY_END---
"""
    return prompt


def parse_exec_brief_response(response_text: str) -> dict:
    """ExecBrief LLM 응답 파싱 → spike/sentiment/opportunity 텍스트 추출"""
    def _extract(text: str, start: str, end: str) -> str:
        s = text.find(start)
        e = text.find(end)
        if s == -1 or e == -1:
            return ""
        return text[s + len(start):e].strip()

    return {
        "spike_summary":       _extract(response_text, "---SPIKE_START---",       "---SPIKE_END---"),
        "sentiment_summary":   _extract(response_text, "---SENTIMENT_START---",   "---SENTIMENT_END---"),
        "opportunity_summary": _extract(response_text, "---OPPORTUNITY_START---", "---OPPORTUNITY_END---"),
    }


# ============================================================
# 추가될 프롬프트 (미구현)
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